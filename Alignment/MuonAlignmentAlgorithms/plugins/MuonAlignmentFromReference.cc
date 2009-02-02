// -*- C++ -*-
//
// Package:    MuonAlignmentAlgorithms
// Class:      MuonAlignmentFromReference
// 
/**\class MuonAlignmentFromReference MuonAlignmentFromReference.cc Alignment/MuonAlignmentFromReference/interface/MuonAlignmentFromReference.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski,,,
//         Created:  Sat Jan 24 16:20:28 CST 2009
//

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"
#include "Alignment/MuonAlignment/interface/AlignableDTSuperLayer.h"
#include "Alignment/MuonAlignment/interface/AlignableDTChamber.h"
#include "Alignment/MuonAlignment/interface/AlignableDTStation.h"
#include "Alignment/MuonAlignment/interface/AlignableDTWheel.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCChamber.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCRing.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCStation.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFromTrack.h"

#include "TFile.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TH1F.h"
#include "TF1.h"

#include <map>
#include <sstream>
#include <fstream>

const double gsbinsize = 0.1;
const double tsbinsize = 0.1;
const int numgsbins = 100;
const int numtsbins = 1000;

Double_t MuonAlignmentFromReference_gpc_function(Double_t *xvec, Double_t *par);
double MuonAlignmentFromReference_lookup_table[numgsbins][numtsbins];

class MuonAlignmentFromReference : public AlignmentAlgorithmBase {
public:
  MuonAlignmentFromReference(const edm::ParameterSet& iConfig);
  ~MuonAlignmentFromReference();
  
  void initialize(const edm::EventSetup& iSetup, AlignableTracker* alignableTracker, AlignableMuon* alignableMuon, AlignmentParameterStore* alignmentParameterStore);
  void startNewLoop();
  void run(const edm::EventSetup& iSetup, const ConstTrajTrackPairCollection& trajtracks);
  void terminate();

  enum {
    kChargeMinus  = 0,
    kChargePlus   = 1,

    kLocalXMinus  = 0,
    kLocalXPlus   = 2,

    kLocalYMinus  = 0,
    kLocalYPlus   = 4,

    kSL13         = 8,
    kSL2          = 16
  };
  
private:
  std::string m_reference;
  double m_minTrackPt;
  int m_minTrackerHits;
  double m_maxTrackerRedChi2;
  bool m_allowTIDTEC;
  int m_minDTSL13Hits;
  int m_minDTSL2Hits;
  double m_maxDTSL13Residual;
  double m_maxDTSL2Residual;
  std::vector<std::string> m_collector;
  std::string m_collectorROOTDir;
  bool m_fitAndAlign;
  int m_minEntriesPerFitBin;
  std::string m_fitReportName;

  AlignableNavigator *m_alignableNavigator;
  AlignmentParameterStore *m_alignmentParameterStore;
  std::vector<Alignable*> m_alignables;

  edm::Service<TFileService> m_tfileService;
  TFileDirectory m_ntupleDirectory, m_fitDirectory;
  std::map<std::pair<Alignable*,unsigned char>,TTree*> m_alignableBin;
  std::vector<TTree*> m_SL13_ttrees, m_SL2_ttrees;
  Float_t m_ttree_x, m_ttree_y, m_ttree_ypos;

  double compute_convolution(double toversigma, double gammaoversigma, double max, double step, double power);
  void initialize_table();
  bool fitbin(TTree *ttree, std::string parameter, double &result, double &uncertainty, double &localy_position);
};

MuonAlignmentFromReference::MuonAlignmentFromReference(const edm::ParameterSet &iConfig)
  : AlignmentAlgorithmBase(iConfig)
  , m_reference(iConfig.getParameter<std::string>("reference"))
  , m_minTrackPt(iConfig.getParameter<double>("minTrackPt"))
  , m_minTrackerHits(iConfig.getParameter<int>("minTrackerHits"))
  , m_maxTrackerRedChi2(iConfig.getParameter<double>("maxTrackerRedChi2"))
  , m_allowTIDTEC(iConfig.getParameter<bool>("allowTIDTEC"))
  , m_minDTSL13Hits(iConfig.getParameter<int>("minDTSL13Hits"))
  , m_minDTSL2Hits(iConfig.getParameter<int>("minDTSL2Hits"))
  , m_maxDTSL13Residual(iConfig.getParameter<double>("maxDTSL13Residual"))
  , m_maxDTSL2Residual(iConfig.getParameter<double>("maxDTSL2Residual"))
  , m_collector(iConfig.getParameter<std::vector<std::string> >("collector"))
  , m_collectorROOTDir(iConfig.getParameter<std::string>("collectorROOTDir"))
  , m_fitAndAlign(iConfig.getParameter<bool>("fitAndAlign"))
  , m_minEntriesPerFitBin(iConfig.getParameter<int>("minEntriesPerFitBin"))
  , m_fitReportName(iConfig.getParameter<std::string>("fitReportName"))
  , m_ntupleDirectory(m_tfileService->mkdir("ntuples"))
  , m_fitDirectory(m_tfileService->mkdir("fits"))
{
}

MuonAlignmentFromReference::~MuonAlignmentFromReference() {
  delete m_alignableNavigator;
}

void MuonAlignmentFromReference::initialize(const edm::EventSetup& iSetup, AlignableTracker* alignableTracker, AlignableMuon* alignableMuon, AlignmentParameterStore* alignmentParameterStore) {
   if (alignableMuon == NULL) {
     throw cms::Exception("MuonAlignmentFromReference") << "doMuon must be set to True" << std::endl;
   }

   m_alignableNavigator = new AlignableNavigator(alignableMuon);
   m_alignmentParameterStore = alignmentParameterStore;
   m_alignables = m_alignmentParameterStore->alignables();

   m_alignableBin.clear();
   m_SL13_ttrees.clear();
   m_SL2_ttrees.clear();
   for (std::vector<Alignable*>::const_iterator ali = m_alignables.begin();  ali != m_alignables.end();  ++ali) {
     if ((*ali)->geomDetId().det() == DetId::Muon  &&  ((*ali)->geomDetId().subdetId() == MuonSubdetId::DT  ||  (*ali)->geomDetId().subdetId() == MuonSubdetId::CSC)) {
       int rawId = (*ali)->id();

       std::stringstream name, title;
       name << rawId;
       if (dynamic_cast<AlignableDetUnit*>(*ali) != NULL  &&  (*ali)->geomDetId().subdetId() == MuonSubdetId::DT) {
	 DTLayerId id(rawId);
	 title << "DT" << id;
       }
       else if (dynamic_cast<AlignableDTSuperLayer*>(*ali) != NULL) {
	 DTSuperLayerId id(rawId);
	 title << "DT" << id;
       }
       else if (dynamic_cast<AlignableDTChamber*>(*ali) != NULL) {
	 DTChamberId id(rawId);
	 title << "DT" << id;
       }
       else if (dynamic_cast<AlignableDTStation*>(*ali) != NULL) {
	 DTChamberId id(rawId);
	 title << "DT Wh:" << id.wheel() << " St:" << id.station() << " ";
       }
       else if (dynamic_cast<AlignableDTWheel*>(*ali) != NULL) {
	 DTChamberId id(rawId);
	 title << "DT Wh:" << id.wheel() << " ";
       }
       else if (dynamic_cast<AlignableDetUnit*>(*ali) != NULL  &&  (*ali)->geomDetId().subdetId() == MuonSubdetId::CSC) {
	 CSCDetId id(rawId);
	 title << "CSC" << id;
       }
       else if (dynamic_cast<AlignableCSCChamber*>(*ali) != NULL) {
	 CSCDetId id(rawId);
	 CSCDetId id2(id.endcap(), id.station(), id.ring(), id.chamber());
	 title << "CSC" << id;
       }
       else if (dynamic_cast<AlignableCSCRing*>(*ali) != NULL) {
	 CSCDetId id(rawId);
	 title << "CSC E:" << id.endcap() << " S:" << id.station() << " R:" << id.ring() << " ";
       }
       else if (dynamic_cast<AlignableCSCStation*>(*ali) != NULL) {
	 CSCDetId id(rawId);
	 title << "CSC E:" << id.endcap() << " S:" << id.station() << " ";
       }
       else {
	 title << "CSC " << rawId << " ";
       }

       for (unsigned char charge = 0;  charge < 2;  charge++) {
	 for (unsigned char localx = 0;  localx < 2;  localx++) {
	   for (unsigned char localy = 0;  localy < 2;  localy++) {
	     std::stringstream myname, mytitle;
	     myname << name.str() << "_q" << (charge == 0 ? "m" : "p") << "_x" << (localx == 0 ? "m" : "p") << "_y" << (localy == 0 ? "m" : "p");
	     mytitle << title.str() << "qbin(" << (charge == 0 ? "-" : "+") << ") xbin(" << (localx == 0 ? "-" : "+") << ") ybin(" << (localy == 0 ? "-" : "+") << ")";

	     if ((*ali)->geomDetId().subdetId() == MuonSubdetId::DT) {
	       std::stringstream myname_sl13, mytitle_sl13;
	       myname_sl13 << myname.str() << "_sl13";
	       mytitle_sl13 << mytitle.str() << ": SL13 residuals";
	       TTree *ttree = m_ntupleDirectory.make<TTree>(myname_sl13.str().c_str(), mytitle_sl13.str().c_str());
	       ttree->Branch("x", &m_ttree_x, "x/F");
	       ttree->Branch("ypos", &m_ttree_ypos, "ypos/F");

  	       m_alignableBin[std::pair<Alignable*,unsigned char>(*ali, kChargePlus*charge + kLocalXPlus*localx + kLocalYPlus*localy + kSL13)] = ttree;
	       m_SL13_ttrees.push_back(ttree);

	       std::stringstream myname_sl2, mytitle_sl2;
	       myname_sl2 << myname.str() << "_sl2";
	       mytitle_sl2 << mytitle.str() << ": SL2 residuals";
	       ttree = m_ntupleDirectory.make<TTree>(myname_sl2.str().c_str(), mytitle_sl2.str().c_str());
	       ttree->Branch("y", &m_ttree_y, "y/F");
	       ttree->Branch("ypos", &m_ttree_ypos, "ypos/F");

 	       m_alignableBin[std::pair<Alignable*,unsigned char>(*ali, kChargePlus*charge + kLocalXPlus*localx + kLocalYPlus*localy + kSL2)] = ttree;
	       m_SL2_ttrees.push_back(ttree);
	     } // end if DT

	     else { // if CSC

	       // later...

	     } // end if CSC

	   } // end localy loop
	 } // end localx loop
       } // end charge loop

     } // end if DT or CSC
   } // end loop over selected alignables

   // now deweight all chambers but the reference

   std::vector<Alignable*> all_DT_chambers = alignableMuon->DTChambers();
   std::vector<Alignable*> all_CSC_chambers = alignableMuon->CSCChambers();
   std::vector<Alignable*> deweight;

   for (std::vector<Alignable*>::const_iterator ali = all_DT_chambers.begin();  ali != all_DT_chambers.end();  ++ali) {
     DTChamberId id((*ali)->geomDetId().rawId());

     if (m_reference == std::string("tracker")) {
       deweight.push_back(*ali);
     }

     else if (m_reference == std::string("wheel0")) {
       if (id.wheel() != 0) {
	 deweight.push_back(*ali);
       }
     }

     else if (m_reference == std::string("wheels0and1")) {
       if (id.wheel() != 0  &&  abs(id.wheel()) != 1) {
	 deweight.push_back(*ali);
       }
     }

     else if (m_reference == std::string("barrel")  ||  m_reference == std::string("me1")  ||  m_reference == std::string("me2")  ||  m_reference == std::string("me3")) {}
   }

   for (std::vector<Alignable*>::const_iterator ali = all_CSC_chambers.begin();  ali != all_CSC_chambers.end();  ++ali) {
     CSCDetId id((*ali)->geomDetId().rawId());

     if (m_reference == std::string("tracker")  ||  m_reference == std::string("wheel0")  ||  m_reference == std::string("wheels0and1")  ||  m_reference == std::string("barrel")) {
       deweight.push_back(*ali);
     }

     else if (m_reference == std::string("me1")) {
       if (id.station() != 1) {
	 deweight.push_back(*ali);
       }
     }

     else if (m_reference == std::string("me2")) {
       if (id.station() != 1  &&  id.station() != 2) {
	 deweight.push_back(*ali);
       }
     }

     else if (m_reference == std::string("me3")) {
       if (id.station() != 1  &&  id.station() != 2  &&  id.station() != 3) {
	 deweight.push_back(*ali);
       }
     }
   }

   alignmentParameterStore->setAlignmentPositionError(deweight, 1000., 0.);
}

void MuonAlignmentFromReference::startNewLoop() {}

void MuonAlignmentFromReference::run(const edm::EventSetup& iSetup, const ConstTrajTrackPairCollection& trajtracks) {
  edm::ESHandle<GlobalTrackingGeometry> globalGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(globalGeometry);

  for (ConstTrajTrackPairCollection::const_iterator trajtrack = trajtracks.begin();  trajtrack != trajtracks.end();  ++trajtrack) {
    const Trajectory* traj = (*trajtrack).first;
    const reco::Track* track = (*trajtrack).second;

    if (track->pt() > m_minTrackPt) {
      MuonResidualsFromTrack muonResidualsFromTrack(globalGeometry, traj, m_alignableNavigator);

      if (muonResidualsFromTrack.trackerNumHits() >= m_minTrackerHits  &&  muonResidualsFromTrack.trackerRedChi2() < m_maxTrackerRedChi2  &&  (m_allowTIDTEC  ||  !muonResidualsFromTrack.contains_TIDTEC())) {

	std::vector<DetId> chamberIds = muonResidualsFromTrack.chamberIds();
	for (std::vector<DetId>::const_iterator chamberId = chamberIds.begin();  chamberId != chamberIds.end();  ++chamberId) {
	  MuonChamberResidual *chamberResidual = muonResidualsFromTrack.chamberResidual(*chamberId);

	  if (chamberId->subdetId() == MuonSubdetId::DT) {
	    if (chamberResidual->isRphiValid()  &&  chamberResidual->rphiHits() >= m_minDTSL13Hits) {
	      unsigned char byte = kSL13;
	      byte += kChargePlus * (track->charge() > 0);
	      byte += kLocalXPlus * (chamberResidual->localx_position(MuonDTChamberResidual::kSuperLayer13) > 0.);
	      byte += kLocalYPlus * (chamberResidual->localy_position(MuonDTChamberResidual::kSuperLayer13) > 0.);
 	      TTree *ttree = m_alignableBin[std::pair<Alignable*,unsigned char>(chamberResidual->chamberAlignable(), byte)];

	      m_ttree_x = chamberResidual->x_residual();
	      m_ttree_ypos = chamberResidual->localy_position(MuonDTChamberResidual::kSuperLayer13);

	      if (fabs(m_ttree_x) < m_maxDTSL13Residual) ttree->Fill();

	    } // end if SL13 is okay

	    if (chamberResidual->isZValid()  &&  chamberResidual->zHits() >= m_minDTSL2Hits) {
	      unsigned char byte = kSL2;
	      byte += kChargePlus * (track->charge() > 0);
	      byte += kLocalXPlus * (chamberResidual->localx_position(MuonDTChamberResidual::kSuperLayer2) > 0.);
	      byte += kLocalYPlus * (chamberResidual->localy_position(MuonDTChamberResidual::kSuperLayer2) > 0.);
 	      TTree *ttree = m_alignableBin[std::pair<Alignable*,unsigned char>(chamberResidual->chamberAlignable(), byte)];

	      m_ttree_y = chamberResidual->y_residual();
	      m_ttree_ypos = chamberResidual->localy_position(MuonDTChamberResidual::kSuperLayer2);
	      if (fabs(m_ttree_y) < m_maxDTSL2Residual) ttree->Fill();
	    } // end if SL2 is okay

	  } // end if DT
      
	  else { // if CSC
	    // later...
	  } // end if CSC

	} // end loop over ids of chambers touched by this track
      } // end if the tracker part of the refit is okay

    } // end if this track satisfies the minimum pT requirement
  } // end loop over trajectory/tracks
}

Double_t MuonAlignmentFromReference_gpc_function(Double_t *xvec, Double_t *par) {
  const Double_t normalization = par[0];
  const Double_t center = par[1];
  const Double_t gamma = fabs(par[2]);
  const Double_t sigma = fabs(par[3]);

  Double_t gammaoversigma = gamma / sigma;
  Double_t toversigma = fabs(xvec[0] - center) / sigma;

  int gsbin0 = int(floor(gammaoversigma / gsbinsize));
  int gsbin1 = int(ceil(gammaoversigma / gsbinsize));
  int tsbin0 = int(floor(toversigma / tsbinsize));
  int tsbin1 = int(ceil(toversigma / tsbinsize));

  if (gsbin0 < 0) gsbin0 = 0;
  if (gsbin1 < 0) gsbin1 = 0;
  if (gsbin0 >= numgsbins) gsbin0 = numgsbins-1;
  if (gsbin1 >= numgsbins) gsbin1 = numgsbins-1;

  if (tsbin0 < 0) tsbin0 = 0;
  if (tsbin1 < 0) tsbin1 = 0;
  if (tsbin0 >= numtsbins) tsbin0 = numtsbins-1;
  if (tsbin1 >= numtsbins) tsbin1 = numtsbins-1;

  Double_t val00 = MuonAlignmentFromReference_lookup_table[gsbin0][tsbin0];
  Double_t val01 = MuonAlignmentFromReference_lookup_table[gsbin0][tsbin1];
  Double_t val10 = MuonAlignmentFromReference_lookup_table[gsbin1][tsbin0];
  Double_t val11 = MuonAlignmentFromReference_lookup_table[gsbin1][tsbin1];

  Double_t val0 = val00 + ((toversigma / tsbinsize) - tsbin0) * (val01 - val00);
  Double_t val1 = val10 + ((toversigma / tsbinsize) - tsbin0) * (val11 - val10);

  Double_t val = val0 + ((gammaoversigma / gsbinsize) - gsbin0) * (val1 - val0);

  return normalization * val / sigma;
}

double MuonAlignmentFromReference::compute_convolution(double toversigma, double gammaoversigma, double max, double step, double power) {
  if (gammaoversigma == 0.) return exp(-toversigma*toversigma/2.) / sqrt(2*M_PI);

  double sum = 0.;
  double uplus = 0.;
  double integrandplus_last = 0.;
  double integrandminus_last = 0.;
  for (double inc = 0.;  uplus < max;  inc += step) {
    double uplus_last = uplus;
    uplus = pow(inc, power);

    double integrandplus = exp(-pow(uplus - toversigma, 2)/2.) / (uplus*uplus/gammaoversigma + gammaoversigma) / M_PI / sqrt(2*M_PI);
    double integrandminus = exp(-pow(-uplus - toversigma, 2)/2.) / (uplus*uplus/gammaoversigma + gammaoversigma) / M_PI / sqrt(2*M_PI);

    sum += integrandplus * (uplus - uplus_last);
    sum += 0.5 * fabs(integrandplus_last - integrandplus) * (uplus - uplus_last);

    sum += integrandminus * (uplus - uplus_last);
    sum += 0.5 * fabs(integrandminus_last - integrandminus) * (uplus - uplus_last);

    integrandplus_last = integrandplus;
    integrandminus_last = integrandminus;
  }
  return sum;
}

void MuonAlignmentFromReference::initialize_table() {
  edm::LogWarning("MuonAlignmentFromReference") << "Initializing convolution look-up table (takes a few minutes)..." << std::endl;
  std::cout << "Initializing convolution look-up table (takes a few minutes)..." << std::endl;

  for (int gsbin = 0;  gsbin < numgsbins;  gsbin++) {
    double gammaoversigma = double(gsbin) * gsbinsize;

    std::cout << "    gsbin " << gsbin << std::endl;

    for (int tsbin = 0;  tsbin < numtsbins;  tsbin++) {
      double toversigma = double(tsbin) * tsbinsize;

      // 1e-6 errors (out of a value of ~0.01) with max=100, step=0.001, power=4
      MuonAlignmentFromReference_lookup_table[gsbin][tsbin] = compute_convolution(toversigma, gammaoversigma, 1000., 0.001, 4.);

      // <10% errors with max=20, step=0.005, power=4
      // MuonAlignmentFromReference_lookup_table[gsbin][tsbin] = compute_convolution(toversigma, gammaoversigma, 100., 0.005, 4.);
    }
  }

  edm::LogWarning("MuonAlignmentFromReference") << "Initialization done!" << std::endl;
  std::cout << "Initialization done!" << std::endl;
}

bool MuonAlignmentFromReference::fitbin(TTree *ttree, std::string parameter, double &result, double &uncertainty, double &localy_position) {
  result = 0.;
  uncertainty = 0.;
  localy_position = 0.;

  // first find the mean and standard deviation
  long numEntries = ttree->GetEntries();
  double sumx = 0.;
  double sumxx = 0.;
  int N = 0;
  for (long i = 0;  i < numEntries;  i++) {
    ttree->GetEntry(i);
    double value = ttree->GetLeaf(parameter.c_str())->GetValue();
    sumx += value;
    sumxx += value * value;
    N++;
  }

  if (N < m_minEntriesPerFitBin) return false;
  double mean = sumx/double(N);
  double stdev = sqrt(sumxx/double(N) - pow(sumx/double(N), 2));

  std::stringstream histname, histtitle, fitname, cut;
  histname << ttree->GetName() << "_" << parameter << "hist";
  histtitle << ttree->GetTitle() << " " << parameter;
  fitname << ttree->GetName() << "_" << parameter << "fit";
  cut << "abs(" << parameter << " - " << mean << ") < " << 5*stdev;

  TF1 *func = new TF1(fitname.str().c_str(), MuonAlignmentFromReference_gpc_function, mean - 5*stdev, mean + 5*stdev, 4);
  func->SetParameters(1, mean, stdev, stdev);
  func->SetParLimits(0, 1, 1);
  func->SetParLimits(1, mean-stdev/2., mean+stdev/2.);
  func->SetParLimits(2, stdev/5., stdev*5.);
  func->SetParLimits(3, stdev/5., stdev*5.);
  
  bool fitSuccessful = (ttree->UnbinnedFit(fitname.str().c_str(), parameter.c_str(), cut.str().c_str(), "QE") == 0);
  if (!fitSuccessful) {
    delete func;
    return false;
  }

  TH1F *hist = m_fitDirectory.make<TH1F>(histname.str().c_str(), histtitle.str().c_str(), 100, mean - 5*stdev, mean + 5*stdev);
  int events_in_window = 0;
  for (long i = 0;  i < numEntries;  i++) {
    ttree->GetEntry(i);
    double value = ttree->GetLeaf(parameter.c_str())->GetValue();
    hist->Fill(value);

    if (mean - 5*stdev < value  &&  value < mean + 5*stdev) {
       localy_position += ttree->GetLeaf("ypos")->GetValue();
       events_in_window++;
    }
  }
  hist->Scale(100. / (10.*stdev) / double(hist->GetEntries()));
  func->Write();

  result = func->GetParameter(1);
  uncertainty = func->GetParError(1);
  localy_position /= double(events_in_window);
  return true;
}

void MuonAlignmentFromReference::terminate() {
  if (m_collector.size() > 0) {
    edm::LogWarning("MuonAlignmentFromReference") << "Attempting to collect ntuples from subjobs into one large file..." << std::endl;
    std::cout << "Attempting to collect ntuples from subjobs into one large file..." << std::endl;
    for (std::vector<TTree*>::const_iterator ttree = m_SL13_ttrees.begin();  ttree != m_SL13_ttrees.end();  ++ttree) {
      for (std::vector<std::string>::const_iterator fileName = m_collector.begin();  fileName != m_collector.end();  ++fileName) {
	std::stringstream pathname;
	pathname << m_collectorROOTDir << "/" << (*ttree)->GetName();

	TFile tfile(fileName->c_str(), "READ");  // do no harm!
	TTree *tmp = (TTree*)(tfile.Get(pathname.str().c_str()));

	// damn... ROOT's TTree::Merge doesn't work, and all the other methods create new TTrees
	// this is surely slower
	long numEntries = tmp->GetEntries();
	for (long i = 0;  i < numEntries;  i++) {
	  tmp->GetEntry(i);
	  m_ttree_x = tmp->GetLeaf("x")->GetValue();
	  m_ttree_ypos = tmp->GetLeaf("ypos")->GetValue();

	  (*ttree)->Fill();
	} // end loop over mini-TTree

	delete tmp;  // this doesn't crash, so I tentatively think I'm supposed to delete it

      } // end loop over subjob fileNames
    } // end loop over SL13 TTrees

    for (std::vector<TTree*>::const_iterator ttree = m_SL2_ttrees.begin();  ttree != m_SL2_ttrees.end();  ++ttree) {
      for (std::vector<std::string>::const_iterator fileName = m_collector.begin();  fileName != m_collector.end();  ++fileName) {
	std::stringstream pathname;
	pathname << m_collectorROOTDir << "/" << (*ttree)->GetName();

	TFile tfile(fileName->c_str(), "READ");  // do no harm!
	TTree *tmp = (TTree*)(tfile.Get(pathname.str().c_str()));

	// damn... ROOT's TTree::Merge doesn't work, and all the other methods create new TTrees
	long numEntries = tmp->GetEntries();
	for (long i = 0;  i < numEntries;  i++) {
	  tmp->GetEntry(i);
	  m_ttree_y = tmp->GetLeaf("y")->GetValue();
	  m_ttree_ypos = tmp->GetLeaf("ypos")->GetValue();

	  (*ttree)->Fill();
	} // end loop over mini-TTree

	delete tmp;  // this doesn't crash, so I tentatively think I'm supposed to delete it

      } // end loop over subjob fileNames
    } // end loop over SL2 TTrees

    edm::LogWarning("MuonAlignmentFromReference") << "Done collecting subjobs!" << std::endl;
    std::cout << "Done collecting subjobs!" << std::endl;
  } // end if we want to collect results from other files

  if (m_fitAndAlign) {
    initialize_table();

    std::ofstream report(m_fitReportName.c_str());

    for (std::vector<Alignable*>::const_iterator ali = m_alignables.begin();  ali != m_alignables.end();  ++ali) {
      std::vector<bool> selector = (*ali)->alignmentParameters()->selector();
      bool align_x = selector[0];
      bool align_y = selector[1];
      bool align_z = selector[2];
      bool align_phix = selector[3];
      bool align_phiy = selector[4];
      bool align_phiz = selector[5];
      int num_params = (align_x + align_y + align_z + align_phix + align_phiy + align_phiz);
      
      if ((*ali)->geomDetId().subdetId() == MuonSubdetId::DT) {
	 bool fitsokay_SL13 = true;
	 double x_correction = 0.;
	 double x_maximalBfieldError = 0.;
	 double x_uncertainty = 0.;
	 double phiz_correction = 0.;
	 double phiz_denominator = 0.;
	 double phiz_uncertainty = 0.;

	 bool fitsokay_SL2 = true;
	 double y_correction = 0.;
	 double y_maximalBfieldError = 0.;
	 double y_uncertainty = 0.;

	 double result, uncertainty, localy_position;

	 for (unsigned char charge = 0;  charge < 2;  charge++) {
	    for (unsigned char localx = 0;  localx < 2;  localx++) {
	       for (unsigned char localy = 0;  localy < 2;  localy++) {
		  TTree *SL13_tree = m_alignableBin[std::pair<Alignable*,unsigned char>(*ali, kChargePlus*charge + kLocalXPlus*localx + kLocalYPlus*localy + kSL13)];
		  TTree *SL2_tree = m_alignableBin[std::pair<Alignable*,unsigned char>(*ali, kChargePlus*charge + kLocalXPlus*localx + kLocalYPlus*localy + kSL2)];

		  if (!fitbin(SL13_tree, std::string("x"), result, uncertainty, localy_position)) fitsokay_SL13 = false;
		  else {
		     x_correction -= result;
		     x_maximalBfieldError -= result * (charge == 1 ? 1. : -1.);
		     x_uncertainty += uncertainty * uncertainty;

		     phiz_correction += result * (localy == 1 ? 1. : -1.);
		     phiz_denominator += localy_position * (localy == 1 ? 1. : -1.);
		     phiz_uncertainty += uncertainty * uncertainty;
		  }

		  if (!fitbin(SL2_tree, std::string("y"), result, uncertainty, localy_position)) fitsokay_SL2 = false;
		  else {
		     y_correction -= result;
		     y_maximalBfieldError -= result * (charge == 1 ? 1. : -1.);
		     y_uncertainty += uncertainty * uncertainty;
		  }

	       } // end localy loop
	    } // end localx loop
	 } // end charge loop

	 DTChamberId chamberId((*ali)->geomDetId().rawId());
	 report << "DT chamber " << chamberId << " (" << chamberId.rawId() << "): " << std::endl;

	 if (fitsokay_SL13) {
	    x_correction /= 8.;
	    x_maximalBfieldError /= 8.;
	    x_uncertainty = sqrt(x_uncertainty) / 8.;
	    phiz_correction /= phiz_denominator;
	    phiz_uncertainty = sqrt(phiz_uncertainty) / phiz_denominator;

	    report << "    successful SL13 fit!" << std::endl;
	    report << "    x: " << x_correction << " +- " << x_uncertainty << " cm " << (align_x ? "(will align)" : "(will NOT align)") << std::endl;
	    report << "    phiz: " << phiz_correction << " +- " << phiz_uncertainty << " rad " << (align_phiz ? "(will align)" : "(will NOT align)") << std::endl;
	    report << "    maximal B-field error (e.g. taking positively-charged tracks only): " << x_maximalBfieldError << std::endl;
	 }

	 if (fitsokay_SL2) {
	    y_correction /= 8.;
	    y_maximalBfieldError /= 8.;
	    y_uncertainty = sqrt(y_uncertainty) / 8.;

	    report << "    successful SL2 fit!" << std::endl;
	    report << "    y: " << y_correction << " +- " << y_uncertainty << " cm " << (align_y ? "(will align)" : "(will NOT align)") << std::endl;
	    report << "    maximal B-field error: " << y_maximalBfieldError << std::endl;
	 }
	    
	 AlgebraicVector params(num_params);
	 AlgebraicSymMatrix cov(num_params);
	 // we don't set {DT,CSC}AlignmentErrorRcds; the uncertainties are for reporting only
	 for (int i = 0;  i < num_params;  i++) {
	    cov[i][i] = 1e-6;
	 }

	 bool update = false;
	 int paramnumber = 0;
	 if (align_x  &&  fitsokay_SL13) {
	    update = true;
	    params[paramnumber] = x_correction;
	    paramnumber++;
	 }
	 if (align_y  &&  fitsokay_SL2) {
	    update = true;
	    params[paramnumber] = y_correction;
	    paramnumber++;
	 }
	 if (align_z) {
	    paramnumber++;
	 }
	 if (align_phix) {
	    paramnumber++;
	 }
	 if (align_phiy) {
	    paramnumber++;
	 }
	 if (align_phiz  &&  fitsokay_SL13) {
	    update = true;
	    params[paramnumber] = phiz_correction;
	    paramnumber++;
	 }

	 if (update) {
	    AlignmentParameters *parnew = (*ali)->alignmentParameters()->cloneFromSelected(params, cov);
	    (*ali)->setAlignmentParameters(parnew);
	    m_alignmentParameterStore->applyParameters(*ali);
	    (*ali)->alignmentParameters()->setValid(true);
	 }
	 report << std::endl;

      } // end if DT
      
      else { // if CSC
	 // later...
      } // end if CSC

    } // end loop over alignables

  } // end if we want to fit the ntuples and align
}

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
DEFINE_EDM_PLUGIN(AlignmentAlgorithmPluginFactory, MuonAlignmentFromReference, "MuonAlignmentFromReference");
