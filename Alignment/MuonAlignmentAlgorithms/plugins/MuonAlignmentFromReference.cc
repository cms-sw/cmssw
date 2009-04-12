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
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TFile.h"

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFromTrack.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsPositionFitter.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsAngleFitter.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsTwoBin.h"

#include <map>
#include <sstream>
#include <fstream>

class MuonAlignmentFromReference : public AlignmentAlgorithmBase {
public:
  MuonAlignmentFromReference(const edm::ParameterSet& iConfig);
  ~MuonAlignmentFromReference();
  
  void initialize(const edm::EventSetup& iSetup, AlignableTracker* alignableTracker, AlignableMuon* alignableMuon, AlignmentParameterStore* alignmentParameterStore);
  void startNewLoop();
  void run(const edm::EventSetup& iSetup, const EventInfo &eventInfo);

  void terminate();

private:
  bool numeric(std::string s);
  int number(std::string s);

  std::vector<std::string> m_intrackfit;
  double m_minTrackPt;
  double m_maxTrackPt;
  int m_minTrackerHits;
  double m_maxTrackerRedChi2;
  bool m_allowTIDTEC;
  int m_minDT13Hits;
  int m_minDT2Hits;
  int m_minCSCHits;
  double m_maxDT13AngleError;
  double m_maxDT2AngleError;
  double m_maxCSCAngleError;
  std::string m_writeTemporaryFile;
  std::vector<std::string> m_readTemporaryFiles;
  bool m_doAlignment;
  std::string m_residualsModel;
  bool m_twoBin;
  bool m_combineME11;
  bool m_DT13fitScattering;
  bool m_DT13fitZpos;
  bool m_DT13fitPhiz;
  bool m_DT2fitScattering;
  bool m_DT2fitPhiz;
  bool m_CSCfitScattering;
  bool m_CSCfitZpos;
  bool m_CSCfitPhiz;
  std::string m_reportFileName;
  std::string m_rootDirectory;

  AlignableNavigator *m_alignableNavigator;
  AlignmentParameterStore *m_alignmentParameterStore;
  std::vector<Alignable*> m_alignables;
  std::map<Alignable*,Alignable*> m_me11map;
  std::map<Alignable*,MuonResidualsTwoBin*> m_rphiFitters;
  std::map<Alignable*,MuonResidualsTwoBin*> m_zFitters;
  std::map<Alignable*,MuonResidualsTwoBin*> m_phixFitters;
  std::map<Alignable*,MuonResidualsTwoBin*> m_phiyFitters;
  std::vector<unsigned int> m_indexOrder;
  std::vector<MuonResidualsTwoBin*> m_fitterOrder;
};

MuonAlignmentFromReference::MuonAlignmentFromReference(const edm::ParameterSet &iConfig)
  : AlignmentAlgorithmBase(iConfig)
  , m_intrackfit(iConfig.getParameter<std::vector<std::string> >("intrackfit"))
  , m_minTrackPt(iConfig.getParameter<double>("minTrackPt"))
  , m_maxTrackPt(iConfig.getParameter<double>("maxTrackPt"))
  , m_minTrackerHits(iConfig.getParameter<int>("minTrackerHits"))
  , m_maxTrackerRedChi2(iConfig.getParameter<double>("maxTrackerRedChi2"))
  , m_allowTIDTEC(iConfig.getParameter<bool>("allowTIDTEC"))
  , m_minDT13Hits(iConfig.getParameter<int>("minDT13Hits"))
  , m_minDT2Hits(iConfig.getParameter<int>("minDT2Hits"))
  , m_minCSCHits(iConfig.getParameter<int>("minCSCHits"))
  , m_maxDT13AngleError(iConfig.getParameter<double>("maxDT13AngleError"))
  , m_maxDT2AngleError(iConfig.getParameter<double>("maxDT2AngleError"))
  , m_maxCSCAngleError(iConfig.getParameter<double>("maxCSCAngleError"))
  , m_writeTemporaryFile(iConfig.getParameter<std::string>("writeTemporaryFile"))
  , m_readTemporaryFiles(iConfig.getParameter<std::vector<std::string> >("readTemporaryFiles"))
  , m_doAlignment(iConfig.getParameter<bool>("doAlignment"))
  , m_residualsModel(iConfig.getParameter<std::string>("residualsModel"))
  , m_twoBin(iConfig.getParameter<bool>("twoBin"))
  , m_combineME11(iConfig.getParameter<bool>("combineME11"))
  , m_DT13fitScattering(iConfig.getParameter<bool>("DT13fitScattering"))
  , m_DT13fitZpos(iConfig.getParameter<bool>("DT13fitZpos"))
  , m_DT13fitPhiz(iConfig.getParameter<bool>("DT13fitPhiz"))
  , m_DT2fitScattering(iConfig.getParameter<bool>("DT2fitScattering"))
  , m_DT2fitPhiz(iConfig.getParameter<bool>("DT2fitPhiz"))
  , m_CSCfitScattering(iConfig.getParameter<bool>("CSCfitScattering"))
  , m_CSCfitZpos(iConfig.getParameter<bool>("CSCfitZpos"))
  , m_CSCfitPhiz(iConfig.getParameter<bool>("CSCfitPhiz"))
  , m_reportFileName(iConfig.getParameter<std::string>("reportFileName"))
  , m_rootDirectory(iConfig.getParameter<std::string>("rootDirectory"))
{
  // alignment requires a TFile to provide plots to check the fit output
  // just filling the residuals lists does not
  // but we don't want to wait until the end of the job to find out that the TFile is missing
  if (m_doAlignment) {
    edm::Service<TFileService> tfileService;
    TFile &tfile = tfileService->file();
    tfile.ls();
  }
}

MuonAlignmentFromReference::~MuonAlignmentFromReference() {
  delete m_alignableNavigator;
}

bool MuonAlignmentFromReference::numeric(std::string s) {
  return (s == std::string("0")  ||  s == std::string("1")  ||  s == std::string("2")  ||  s == std::string("3")  ||  s == std::string("4")  ||
	  s == std::string("5")  ||  s == std::string("6")  ||  s == std::string("7")  ||  s == std::string("8")  ||  s == std::string("9"));
}

int MuonAlignmentFromReference::number(std::string s) {
  if (s == std::string("0")) return 0;
  else if (s == std::string("1")) return 1;
  else if (s == std::string("2")) return 2;
  else if (s == std::string("3")) return 3;
  else if (s == std::string("4")) return 4;
  else if (s == std::string("5")) return 5;
  else if (s == std::string("6")) return 6;
  else if (s == std::string("7")) return 7;
  else if (s == std::string("8")) return 8;
  else if (s == std::string("9")) return 9;
  else assert(false);
}

void MuonAlignmentFromReference::initialize(const edm::EventSetup& iSetup, AlignableTracker* alignableTracker, AlignableMuon* alignableMuon, AlignmentParameterStore* alignmentParameterStore) {
   if (alignableMuon == NULL) {
     throw cms::Exception("MuonAlignmentFromReference") << "doMuon must be set to True" << std::endl;
   }

   m_alignableNavigator = new AlignableNavigator(alignableMuon);
   m_alignmentParameterStore = alignmentParameterStore;
   m_alignables = m_alignmentParameterStore->alignables();

   int residualsModel;
   if (m_residualsModel == std::string("pureGaussian")) residualsModel = MuonResidualsFitter::kPureGaussian;
   else if (m_residualsModel == std::string("powerLawTails")) residualsModel = MuonResidualsFitter::kPowerLawTails;
   else throw cms::Exception("MuonAlignmentFromReference") << "unrecognized residualsModel: \"" << m_residualsModel << "\"" << std::endl;

   // set up the MuonResidualsFitters (which also collect residuals for fitting)
   m_me11map.clear();
   m_rphiFitters.clear();
   m_zFitters.clear();
   m_phixFitters.clear();
   m_phiyFitters.clear();
   m_indexOrder.clear();
   m_fitterOrder.clear();
   for (std::vector<Alignable*>::const_iterator ali = m_alignables.begin();  ali != m_alignables.end();  ++ali) {
     std::vector<bool> selector = (*ali)->alignmentParameters()->selector();
     bool align_x = selector[0];
     bool align_y = selector[1];
     //     bool align_z = selector[2];
     bool align_phix = selector[3];
     //     bool align_phiy = selector[4];
     bool align_phiz = selector[5];

     if ((*ali)->alignableObjectId() == align::AlignableDTChamber) {
       m_rphiFitters[*ali] = new MuonResidualsTwoBin(m_twoBin, new MuonResidualsPositionFitter(residualsModel, 5), new MuonResidualsPositionFitter(residualsModel, 5));
       if (!m_DT13fitScattering) m_rphiFitters[*ali]->fix(MuonResidualsPositionFitter::kScattering);
       if (!m_DT13fitZpos) m_rphiFitters[*ali]->fix(MuonResidualsPositionFitter::kZpos);
       if (!m_DT13fitPhiz) m_rphiFitters[*ali]->fix(MuonResidualsPositionFitter::kPhiz);
       m_indexOrder.push_back((*ali)->geomDetId().rawId()*4 + 0);
       m_fitterOrder.push_back(m_rphiFitters[*ali]);
       
       m_zFitters[*ali] = new MuonResidualsTwoBin(m_twoBin, new MuonResidualsPositionFitter(residualsModel, 5), new MuonResidualsPositionFitter(residualsModel, 5));
       if (!m_DT2fitScattering) m_zFitters[*ali]->fix(MuonResidualsPositionFitter::kScattering);
       m_zFitters[*ali]->fix(MuonResidualsPositionFitter::kZpos);
       if (!m_DT2fitPhiz) m_zFitters[*ali]->fix(MuonResidualsPositionFitter::kPhiz);
       m_indexOrder.push_back((*ali)->geomDetId().rawId()*4 + 1);
       m_fitterOrder.push_back(m_zFitters[*ali]);

       m_phixFitters[*ali] = new MuonResidualsTwoBin(m_twoBin, new MuonResidualsAngleFitter(residualsModel, 5), new MuonResidualsAngleFitter(residualsModel, 5));
       m_indexOrder.push_back((*ali)->geomDetId().rawId()*4 + 2);
       m_fitterOrder.push_back(m_phixFitters[*ali]);

       m_phiyFitters[*ali] = new MuonResidualsTwoBin(m_twoBin, new MuonResidualsAngleFitter(residualsModel, 5), new MuonResidualsAngleFitter(residualsModel, 5));
       m_indexOrder.push_back((*ali)->geomDetId().rawId()*4 + 3);
       m_fitterOrder.push_back(m_phiyFitters[*ali]);
     }

     else if ((*ali)->alignableObjectId() == align::AlignableCSCChamber) {
       if (align_x  &&  (!align_y  ||  !align_phiz)) throw cms::Exception("MuonAlignmentFromReference") << "CSCs are aligned in rphi, not x, so y and phiz must also be alignable" << std::endl;

       Alignable *thisali = *ali;
       CSCDetId id((*ali)->geomDetId().rawId());
       if (m_combineME11  &&  id.station() == 1  &&  id.ring() == 4) {
	 CSCDetId pairid(id.endcap(), 1, 1, id.chamber());
	 
	 for (std::vector<Alignable*>::const_iterator ali2 = m_alignables.begin();  ali2 != m_alignables.end();  ++ali2) {
	   if ((*ali2)->alignableObjectId() == align::AlignableCSCChamber  &&  (*ali2)->geomDetId().rawId() == pairid.rawId()) {
	     thisali = *ali2;
	     break;
	   }
	 }

	 m_me11map[*ali] = thisali;
       }

       if (thisali == *ali) {

	 m_rphiFitters[*ali] = new MuonResidualsTwoBin(m_twoBin, new MuonResidualsPositionFitter(residualsModel, 5), new MuonResidualsPositionFitter(residualsModel, 5));
	 if (!m_CSCfitScattering) m_rphiFitters[*ali]->fix(MuonResidualsPositionFitter::kScattering);
	 if (!m_CSCfitZpos) m_rphiFitters[*ali]->fix(MuonResidualsPositionFitter::kZpos);
	 if (!m_CSCfitPhiz) m_rphiFitters[*ali]->fix(MuonResidualsPositionFitter::kPhiz);
	 m_indexOrder.push_back((*ali)->geomDetId().rawId()*4 + 0);
	 m_fitterOrder.push_back(m_rphiFitters[*ali]);
	 
	 m_phiyFitters[*ali] = new MuonResidualsTwoBin(m_twoBin, new MuonResidualsAngleFitter(residualsModel, 5), new MuonResidualsAngleFitter(residualsModel, 5));
	 m_indexOrder.push_back((*ali)->geomDetId().rawId()*4 + 1);
	 m_fitterOrder.push_back(m_phiyFitters[*ali]);
	 
	 if (align_phix) {
	   throw cms::Exception("MuonAlignmentFromReference") << "CSCChambers can't be aligned in phix" << std::endl;
	 }

       }
     }

     else {
       throw cms::Exception("MuonAlignmentFromReference") << "only DTChambers and CSCChambers are alignable" << std::endl;
     }
   } // end loop over chambers chosen for alignment

   // deweight all chambers but the reference
   std::vector<Alignable*> all_DT_chambers = alignableMuon->DTChambers();
   std::vector<Alignable*> all_CSC_chambers = alignableMuon->CSCChambers();
   std::vector<Alignable*> intrackfit;
   std::map<Alignable*,bool> already_seen;

   for (std::vector<std::string>::const_iterator name = m_intrackfit.begin();  name != m_intrackfit.end();  ++name) {
     bool parsing_error = false;

     bool barrel = (name->substr(0, 2) == std::string("MB"));
     bool endcap = (name->substr(0, 2) == std::string("ME"));
     if (!barrel  &&  !endcap) parsing_error = true;

     if (!parsing_error  &&  barrel) {
       int index = 2;

       if (name->substr(index, 1) == std::string(" ")) {
	 index++;
       }

       bool plus = true;
       if (name->substr(index, 1) == std::string("+")) {
	 plus = true;
	 index++;
       }
       else if (name->substr(index, 1) == std::string("-")) {
	 plus = false;
	 index++;
       }
       else if (numeric(name->substr(index, 1))) {}
       else parsing_error = true;

       int wheel = 0;
       bool wheel_digit = false;
       while (!parsing_error  &&  numeric(name->substr(index, 1))) {
	 wheel *= 10;
	 wheel += number(name->substr(index, 1));
	 wheel_digit = true;
	 index++;
       }
       if (!plus) wheel *= -1;
       if (!wheel_digit) parsing_error = true;
       
       if (name->substr(index, 1) != std::string(" ")) parsing_error = true;
       index++;

       int station = 0;
       bool station_digit = false;
       while (!parsing_error  &&  numeric(name->substr(index, 1))) {
	 station *= 10;
	 station += number(name->substr(index, 1));
	 station_digit = true;
	 index++;
       }
       if (!station_digit) parsing_error = true;

       if (name->substr(index, 1) != std::string(" ")) parsing_error = true;
       index++;

       int sector = 0;
       bool sector_digit = false;
       while (!parsing_error  &&  numeric(name->substr(index, 1))) {
	 sector *= 10;
	 sector += number(name->substr(index, 1));
	 sector_digit = true;
	 index++;
       }
       if (!sector_digit) parsing_error = true;

       if (!parsing_error) {
	 bool no_such_chamber = false;

	 if (wheel < -2  ||  wheel > 2) no_such_chamber = true;
	 if (station < 1  ||  station > 4) no_such_chamber = true;
	 if (station == 4  &&  (sector < 1  ||  sector > 14)) no_such_chamber = true;
	 if (station < 4  &&  (sector < 1  ||  sector > 12)) no_such_chamber = true;

	 if (no_such_chamber) {
	   throw cms::Exception("MuonAlignmentFromReference") << "intrackfit chamber doesn't exist: " << (*name) << std::endl;
	 }

	 DTChamberId id(wheel, station, sector);
	 for (std::vector<Alignable*>::const_iterator ali = all_DT_chambers.begin();  ali != all_DT_chambers.end();  ++ali) {
	   if ((*ali)->geomDetId().rawId() == id.rawId()) {
	     std::map<Alignable*,bool>::const_iterator trial = already_seen.find(*ali);
	     if (trial == already_seen.end()) {
	       intrackfit.push_back(*ali);
	       already_seen[*ali] = true;
	     }
	   }
	 }
       }
     }
     if (!parsing_error  &&  endcap) {
       int index = 2;

       if (name->substr(index, 1) == std::string(" ")) {
	 index++;
       }

       bool plus = true;
       if (name->substr(index, 1) == std::string("+")) {
	 plus = true;
	 index++;
       }
       else if (name->substr(index, 1) == std::string("-")) {
	 plus = false;
	 index++;
       }
       else if (numeric(name->substr(index, 1))) {}
       else parsing_error = true;

       int station = 0;
       bool station_digit = false;
       while (!parsing_error  &&  numeric(name->substr(index, 1))) {
	 station *= 10;
	 station += number(name->substr(index, 1));
	 station_digit = true;
	 index++;
       }
       if (!plus) station *= -1;
       if (!station_digit) parsing_error = true;

       if (name->substr(index, 1) != std::string("/")) parsing_error = true;
       index++;

       int ring = 0;
       bool ring_digit = false;
       while (!parsing_error  &&  numeric(name->substr(index, 1))) {
	 ring *= 10;
	 ring += number(name->substr(index, 1));
	 ring_digit = true;
	 index++;
       }
       if (!ring_digit) parsing_error = true;

       if (name->substr(index, 1) != std::string(" ")) parsing_error = true;
       index++;

       int chamber = 0;
       bool chamber_digit = false;
       while (!parsing_error  &&  numeric(name->substr(index, 1))) {
	 chamber *= 10;
	 chamber += number(name->substr(index, 1));
	 chamber_digit = true;
	 index++;
       }
       if (!chamber_digit) parsing_error = true;

       if (!parsing_error) {
	 bool no_such_chamber = false;

	 int endcap = (station > 0 ? 1 : 2);
	 station = abs(station);
	 if (station < 1  ||  station > 4) no_such_chamber = true;
	 if (station == 1  &&  (ring < 1  ||  ring > 4)) no_such_chamber = true;
	 if (station > 1  &&  (ring < 1  ||  ring > 2)) no_such_chamber = true;
	 if (station == 1  &&  (chamber < 1  ||  chamber > 36)) no_such_chamber = true;
	 if (station > 1  &&  ring == 1  &&  (chamber < 1  ||  chamber > 18)) no_such_chamber = true;
	 if (station > 1  &&  ring == 2  &&  (chamber < 1  ||  chamber > 36)) no_such_chamber = true;

	 if (no_such_chamber) {
	   throw cms::Exception("MuonAlignmentFromReference") << "intrackfit chamber doesn't exist: " << (*name) << std::endl;
	 }

	 CSCDetId id(endcap, station, ring, chamber);
	 for (std::vector<Alignable*>::const_iterator ali = all_CSC_chambers.begin();  ali != all_CSC_chambers.end();  ++ali) {
	   if ((*ali)->geomDetId().rawId() == id.rawId()) {
	     std::map<Alignable*,bool>::const_iterator trial = already_seen.find(*ali);
	     if (trial == already_seen.end()) {
	       intrackfit.push_back(*ali);
	       already_seen[*ali] = true;
	     }
	   }
	 }
       }
     }

     if (parsing_error) {
       throw cms::Exception("MuonAlignmentFromReference") << "intrackfit chamber name is malformed: " << (*name) << std::endl;
     }
   }

   alignmentParameterStore->setAlignmentPositionError(all_DT_chambers, 1000., 0.);
   alignmentParameterStore->setAlignmentPositionError(all_CSC_chambers, 1000., 0.);
   alignmentParameterStore->setAlignmentPositionError(intrackfit, 0., 0.);
}

void MuonAlignmentFromReference::startNewLoop() {}

void MuonAlignmentFromReference::run(const edm::EventSetup& iSetup, const EventInfo &eventInfo)
{

  edm::ESHandle<GlobalTrackingGeometry> globalGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(globalGeometry);

  const ConstTrajTrackPairCollection &trajtracks = eventInfo.trajTrackPairs_;
  for (ConstTrajTrackPairCollection::const_iterator trajtrack = trajtracks.begin();  trajtrack != trajtracks.end();  ++trajtrack) {
    const Trajectory* traj = (*trajtrack).first;
    const reco::Track* track = (*trajtrack).second;

    if (m_minTrackPt < track->pt()  &&  track->pt() < m_maxTrackPt) {
      char charge = (track->charge() > 0 ? 1 : -1);
      // double qoverpt = track->charge() / track->pt();
      // double qoverpz = track->charge() / track->pz();
      MuonResidualsFromTrack muonResidualsFromTrack(globalGeometry, traj, m_alignableNavigator, 1000.);

      if (muonResidualsFromTrack.trackerNumHits() >= m_minTrackerHits  &&  muonResidualsFromTrack.trackerRedChi2() < m_maxTrackerRedChi2  &&  (m_allowTIDTEC  ||  !muonResidualsFromTrack.contains_TIDTEC())) {
	std::vector<unsigned int> indexes = muonResidualsFromTrack.indexes();

	for (std::vector<unsigned int>::const_iterator index = indexes.begin();  index != indexes.end();  ++index) {
	  MuonChamberResidual *chamberResidual = muonResidualsFromTrack.chamberResidual(*index);

	  if (chamberResidual->chamberId().subdetId() == MuonSubdetId::DT  &&  (*index) % 2 == 0) {

	    if (chamberResidual->numHits() >= m_minDT13Hits) {
	      std::map<Alignable*,MuonResidualsTwoBin*>::const_iterator rphiFitter = m_rphiFitters.find(chamberResidual->chamberAlignable());
	      std::map<Alignable*,MuonResidualsTwoBin*>::const_iterator phiyFitter = m_phiyFitters.find(chamberResidual->chamberAlignable());

	      if (rphiFitter != m_rphiFitters.end()) {
		if (fabs(chamberResidual->resslope()) < m_maxDT13AngleError) {
		  double *residdata = new double[MuonResidualsPositionFitter::kNData];
		  residdata[MuonResidualsPositionFitter::kResidual] = chamberResidual->residual();
		  residdata[MuonResidualsPositionFitter::kAngleError] = chamberResidual->resslope();
		  residdata[MuonResidualsPositionFitter::kTrackAngle] = chamberResidual->trackdxdz();
		  residdata[MuonResidualsPositionFitter::kTrackPosition] = chamberResidual->tracky();
		  rphiFitter->second->fill(charge, residdata);
		  // the MuonResidualsPositionFitter will delete the array when it is destroyed
		}
	      }
	      
	      if (phiyFitter != m_phiyFitters.end()) {
		double *residdata = new double[MuonResidualsAngleFitter::kNData];
		residdata[MuonResidualsAngleFitter::kResidual] = chamberResidual->resslope();
		residdata[MuonResidualsAngleFitter::kXPosition] = chamberResidual->trackx();
		residdata[MuonResidualsAngleFitter::kYPosition] = chamberResidual->tracky();
		phiyFitter->second->fill(charge, residdata);
		// the MuonResidualsAngleFitter will delete the array when it is destroyed
	      }
	    }
	  } // end if DT13

	  else if (chamberResidual->chamberId().subdetId() == MuonSubdetId::DT  &&  (*index) % 2 == 1) {
	    if (chamberResidual->numHits() >= m_minDT2Hits) {
	      std::map<Alignable*,MuonResidualsTwoBin*>::const_iterator zFitter = m_zFitters.find(chamberResidual->chamberAlignable());
	      std::map<Alignable*,MuonResidualsTwoBin*>::const_iterator phixFitter = m_phixFitters.find(chamberResidual->chamberAlignable());

	      if (zFitter != m_zFitters.end()) {
		if (fabs(chamberResidual->resslope()) < m_maxDT2AngleError) {
		  double *residdata = new double[MuonResidualsPositionFitter::kNData];
		  residdata[MuonResidualsPositionFitter::kResidual] = chamberResidual->residual();
		  residdata[MuonResidualsPositionFitter::kAngleError] = chamberResidual->resslope();
		  residdata[MuonResidualsPositionFitter::kTrackAngle] = chamberResidual->trackdydz();
		  residdata[MuonResidualsPositionFitter::kTrackPosition] = chamberResidual->trackx();
		  zFitter->second->fill(charge, residdata);
		  // the MuonResidualsPositionFitter will delete the array when it is destroyed
		}
	      }

	      if (phixFitter != m_phixFitters.end()) {
		double *residdata = new double[MuonResidualsAngleFitter::kNData];
		residdata[MuonResidualsAngleFitter::kResidual] = chamberResidual->resslope();
		residdata[MuonResidualsAngleFitter::kXPosition] = chamberResidual->trackx();
		residdata[MuonResidualsAngleFitter::kYPosition] = chamberResidual->tracky();
		phixFitter->second->fill(charge, residdata);
		// the MuonResidualsAngleFitter will delete the array when it is destroyed
	      }
	    }
	  } // end if DT2

	  else if (chamberResidual->chamberId().subdetId() == MuonSubdetId::CSC) {

	    if (chamberResidual->numHits() >= m_minCSCHits) {
	      Alignable *ali = chamberResidual->chamberAlignable();
	      CSCDetId id(ali->geomDetId().rawId());
	      if (m_combineME11  &&  id.station() == 1  &&  id.ring() == 4) {
		ali = m_me11map[ali];
	      }

	      std::map<Alignable*,MuonResidualsTwoBin*>::const_iterator rphiFitter = m_rphiFitters.find(ali);
	      std::map<Alignable*,MuonResidualsTwoBin*>::const_iterator phiyFitter = m_phiyFitters.find(ali);

	      if (rphiFitter != m_rphiFitters.end()) {
		if (fabs(chamberResidual->resslope()) < m_maxCSCAngleError) {
		  double *residdata = new double[MuonResidualsPositionFitter::kNData];
		  residdata[MuonResidualsPositionFitter::kResidual] = chamberResidual->residual();
		  residdata[MuonResidualsPositionFitter::kAngleError] = chamberResidual->resslope();
		  residdata[MuonResidualsPositionFitter::kTrackAngle] = chamberResidual->trackdxdz();
		  residdata[MuonResidualsPositionFitter::kTrackPosition] = chamberResidual->tracky();
		  rphiFitter->second->fill(charge, residdata);
		  // the MuonResidualsPositionFitter will delete the array when it is destroyed
		}
	      }

	      if (phiyFitter != m_phiyFitters.end()) {
		double *residdata = new double[MuonResidualsAngleFitter::kNData];
		residdata[MuonResidualsAngleFitter::kResidual] = chamberResidual->resslope();
		residdata[MuonResidualsAngleFitter::kXPosition] = chamberResidual->trackx();
		residdata[MuonResidualsAngleFitter::kYPosition] = chamberResidual->tracky();
		phiyFitter->second->fill(charge, residdata);
		// the MuonResidualsAngleFitter will delete the array when it is destroyed
	      }
	    }
	  } // end if CSC

	} // end loop over chamberIds
      } // end if refit is okay
    } // end if track pT is within range
  } // end loop over tracks
}

void MuonAlignmentFromReference::terminate() {
  // collect temporary files
  if (m_readTemporaryFiles.size() != 0) {
    for (std::vector<std::string>::const_iterator fileName = m_readTemporaryFiles.begin();  fileName != m_readTemporaryFiles.end();  ++fileName) {
      FILE *file;
      int size;
      file = fopen(fileName->c_str(), "r");
      fread(&size, sizeof(int), 1, file);
      if (int(m_indexOrder.size()) != size) throw cms::Exception("MuonAlignmentFromReference") << "file \"" << *fileName << "\" has " << size << " fitters, but this job has " << m_indexOrder.size() << " fitters (probably corresponds to the wrong alignment job)" << std::endl;
      
      std::vector<unsigned int>::const_iterator index = m_indexOrder.begin();
      std::vector<MuonResidualsTwoBin*>::const_iterator fitter = m_fitterOrder.begin();
      for (int i = 0;  i < size;  ++i, ++index, ++fitter) {
	unsigned int index_toread;
	fread(&index_toread, sizeof(unsigned int), 1, file);
	if (*index != index_toread) throw cms::Exception("MuonAlignmentFromReference") << "file \"" << *fileName << "\" has index " << index_toread << " at position " << i << ", but this job is expecting " << *index << " (probably corresponds to the wrong alignment job)" << std::endl;
	(*fitter)->read(file, i);
      }

      fclose(file);
    }
  }

  // fit and align (time-consuming, so the user can turn it off if in
  // a residuals-gathering job)
  if (m_doAlignment) {
    edm::Service<TFileService> tfileService;
    TFileDirectory rootDirectory(m_rootDirectory == std::string("") ? *tfileService : tfileService->mkdir(m_rootDirectory));

    std::ofstream report;
    bool writeReport = (m_reportFileName != std::string(""));
    if (writeReport) {
      report.open(m_reportFileName.c_str());
      report << "reports = []" << std::endl;
      report << "class Report:" << std::endl
	     << "    def __init__(self, chamberId, postal_address, name):" << std::endl
	     << "        self.chamberId, self.postal_address, self.name = chamberId, postal_address, name" << std::endl
	     << "        self.phiyFit_status = \"UNKNOWN\"" << std::endl
	     << "        self.rphiFit_status = \"UNKNOWN\"" << std::endl
	     << "        self.phixFit_status = \"UNKNOWN\"" << std::endl
	     << "        self.zFit_status = \"UNKNOWN\"" << std::endl
	     << "" << std::endl
	     << "    def phiyFit(self, angle, sigma, gamma, redchi2, posNum, negNum):" << std::endl
	     << "        self.phiyFit_status = \"PASS\"" << std::endl
	     << "        self.phiyFit_angle = angle" << std::endl
	     << "        self.phiyFit_sigma = sigma" << std::endl
	     << "        self.phiyFit_gamma = gamma" << std::endl
	     << "        self.phiyFit_redchi2 = redchi2" << std::endl
	     << "        self.phiyFit_posNum = posNum" << std::endl
	     << "        self.phiyFit_negNum = negNum" << std::endl
	     << "" << std::endl
	     << "    def rphiFit(self, position, zpos, phiz, scattering, sigma, gamma, redchi2, posNum, negNum):" << std::endl
	     << "        self.rphiFit_status = \"PASS\"" << std::endl
	     << "        self.rphiFit_position = position" << std::endl
	     << "        self.rphiFit_zpos = zpos" << std::endl
	     << "        self.rphiFit_phiz = phiz" << std::endl
	     << "        self.rphiFit_scattering = scattering" << std::endl
	     << "        self.rphiFit_sigma = sigma" << std::endl
	     << "        self.rphiFit_gamma = gamma" << std::endl
	     << "        self.rphiFit_redchi2 = redchi2" << std::endl
	     << "        self.rphiFit_posNum = posNum" << std::endl
	     << "        self.rphiFit_negNum = negNum" << std::endl
	     << "" << std::endl
	     << "    def phixFit(self, angle, sigma, gamma, redchi2, posNum, negNum):" << std::endl
	     << "        self.phixFit_status = \"PASS\"" << std::endl
	     << "        self.phixFit_angle = angle" << std::endl
	     << "        self.phixFit_sigma = sigma" << std::endl
	     << "        self.phixFit_gamma = gamma" << std::endl
	     << "        self.phixFit_redchi2 = redchi2" << std::endl
	     << "        self.phixFit_posNum = posNum" << std::endl
	     << "        self.phixFit_negNum = negNum" << std::endl
	     << "" << std::endl
	     << "    def zFit(self, position, zpos, phiz, scattering, sigma, gamma, redchi2, posNum, negNum):" << std::endl
	     << "        self.zFit_status = \"PASS\"" << std::endl
	     << "        self.zFit_position = position" << std::endl
	     << "        self.zFit_zpos = zpos" << std::endl
	     << "        self.zFit_phiz = phiz" << std::endl
	     << "        self.zFit_scattering = scattering" << std::endl
	     << "        self.zFit_sigma = sigma" << std::endl
	     << "        self.zFit_gamma = gamma" << std::endl
	     << "        self.zFit_redchi2 = redchi2" << std::endl
	     << "        self.zFit_posNum = posNum" << std::endl
	     << "        self.zFit_negNum = negNum" << std::endl
	     << "" << std::endl
	     << "    def parameters(self, deltax, deltay, deltaz, deltaphix, deltaphiy, deltaphiz):" << std::endl
	     << "        self.deltax, self.deltay, self.deltaz, self.deltaphix, self.deltaphiy, self.deltaphiz = \\" << std::endl
	     << "                     deltax, deltay, deltaz, deltaphix, deltaphiy, deltaphiz" << std::endl
	     << "" << std::endl
	     << "    def errors(self, err2x, err2y, err2z):" << std::endl
	     << "        self.err2x, self.err2y, self.err2z = err2x, err2y, err2z" << std::endl << std::endl << std::endl;
    }
    
    for (std::vector<Alignable*>::const_iterator ali = m_alignables.begin();  ali != m_alignables.end();  ++ali) {
      std::vector<bool> selector = (*ali)->alignmentParameters()->selector();
      bool align_x = selector[0];
      bool align_y = selector[1];
      bool align_z = selector[2];
      bool align_phix = selector[3];
      bool align_phiy = selector[4];
      bool align_phiz = selector[5];
      int numParams = ((align_x ? 1 : 0) + (align_y ? 1 : 0) + (align_z ? 1 : 0) + (align_phix ? 1 : 0) + (align_phiy ? 1 : 0) + (align_phiz ? 1 : 0));

      // map from 0-5 to the index of params, above
      std::vector<int> paramIndex;
      int paramIndex_counter = -1;
      if (align_x) paramIndex_counter++;
      paramIndex.push_back(paramIndex_counter);
      if (align_y) paramIndex_counter++;
      paramIndex.push_back(paramIndex_counter);
      if (align_z) paramIndex_counter++;
      paramIndex.push_back(paramIndex_counter);
      if (align_phix) paramIndex_counter++;
      paramIndex.push_back(paramIndex_counter);
      if (align_phiy) paramIndex_counter++;
      paramIndex.push_back(paramIndex_counter);
      if (align_phiz) paramIndex_counter++;
      paramIndex.push_back(paramIndex_counter);

      // uncertainties will be infinite except for the aligned chambers in the aligned directions
      AlgebraicVector params(numParams);
      AlgebraicSymMatrix cov(numParams);
      for (int i = 0;  i < numParams;  i++) {
	for (int j = 0;  j < numParams;  j++) {
	  cov[i][j] = 0.;
	}
	params[i] = 0.;
      }
      // but the translational ones only, because that's all that's stored
      cov[paramIndex[0]][paramIndex[0]] = 1000.;
      cov[paramIndex[1]][paramIndex[1]] = 1000.;
      cov[paramIndex[2]][paramIndex[2]] = 1000.;

      DetId id = (*ali)->geomDetId();

      Alignable *thisali = *ali;
      if (m_combineME11  &&  id.subdetId() == MuonSubdetId::CSC) {
	CSCDetId cscid(id.rawId());
	if (cscid.station() == 1  &&  cscid.ring() == 4) {
	  thisali = m_me11map[*ali];
	}
      }

      std::map<Alignable*,MuonResidualsTwoBin*>::const_iterator rphiFitter = m_rphiFitters.find(thisali);
      std::map<Alignable*,MuonResidualsTwoBin*>::const_iterator zFitter = m_zFitters.find(thisali);
      std::map<Alignable*,MuonResidualsTwoBin*>::const_iterator phixFitter = m_phixFitters.find(thisali);
      std::map<Alignable*,MuonResidualsTwoBin*>::const_iterator phiyFitter = m_phiyFitters.find(thisali);
      
      std::stringstream name;
      if (id.subdetId() == MuonSubdetId::DT) {
	DTChamberId chamberId(id.rawId());
	name << "MBwh";
	if (chamberId.wheel() == -2) name << "A";
	else if (chamberId.wheel() == -1) name << "B";
	else if (chamberId.wheel() ==  0) name << "C";
	else if (chamberId.wheel() == +1) name << "D";
	else if (chamberId.wheel() == +2) name << "E";
	std::string sectoro("0");
	if (chamberId.sector() > 9) sectoro = std::string("");
	name << "st" << chamberId.station() << "sec" << sectoro << chamberId.sector();

	if (writeReport) {
	  report << "reports.append(Report(" << id.rawId() << ", (\"DT\", " << chamberId.wheel() << ", " << chamberId.station() << ", " << chamberId.sector() << "), \"" << name.str() << "\"))" << std::endl;
	}
      }
      else if (id.subdetId() == MuonSubdetId::CSC) {
	CSCDetId chamberId(id.rawId());
	std::string chambero("0");
	if (chamberId.chamber() > 9) chambero = std::string("");
	name << "ME" << (chamberId.endcap() == 1 ? "p" : "m") << abs(chamberId.station()) << chamberId.ring() << "_" << chambero << chamberId.chamber();

	if (writeReport) {
	  report << "reports.append(Report(" << id.rawId() << ", (\"CSC\", " << (chamberId.endcap() == 1 ? 1 : -1)*abs(chamberId.station()) << ", " << chamberId.ring() << ", " << chamberId.chamber() << "), \"" << name.str() << "\"))" << std::endl;
	}
      }

      bool phiyOkay = false;
      double phiyValue = 0.;
      if (phiyFitter != m_phiyFitters.end()) {
	// the fit is verbose in std::cout anyway
	std::cout << "=============================================================================================" << std::endl;
	std::cout << "Fitting " << name.str() << " phiy" << std::endl;

	if (phiyFitter->second->fit(0.)) {
	  std::stringstream name2;
	  name2 << name.str() << "_phiyFit";
	  phiyFitter->second->plot(0., name2.str(), &rootDirectory);
	  double redchi2 = phiyFitter->second->redchi2(0., name2.str(), &rootDirectory);
	  long posNum = phiyFitter->second->numResidualsPos();
	  long negNum = phiyFitter->second->numResidualsNeg();

	  double angle_value = phiyFitter->second->value(MuonResidualsAngleFitter::kAngle);
	  double angle_error = phiyFitter->second->error(MuonResidualsAngleFitter::kAngle);
	  double angle_antisym = phiyFitter->second->antisym(MuonResidualsAngleFitter::kAngle);
	  phiyOkay = true;
	  phiyValue = angle_value;

	  double sigma_value = phiyFitter->second->value(MuonResidualsAngleFitter::kSigma);
	  double sigma_error = phiyFitter->second->error(MuonResidualsAngleFitter::kSigma);
	  double sigma_antisym = phiyFitter->second->antisym(MuonResidualsAngleFitter::kSigma);

	  double gamma_value, gamma_error, gamma_antisym;
	  gamma_value = gamma_error = gamma_antisym = 0.;
	  if (phiyFitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian) {
	    gamma_value = phiyFitter->second->value(MuonResidualsAngleFitter::kGamma);
	    gamma_error = phiyFitter->second->error(MuonResidualsAngleFitter::kGamma);
	    gamma_antisym = phiyFitter->second->antisym(MuonResidualsAngleFitter::kGamma);
	  }

	  if (id.subdetId() == MuonSubdetId::DT) {
	    if (align_phiy) {
	      params[paramIndex[4]] = angle_value;
	    }
	  } // end if DT

	  else {
	    if (align_phiy) {
	      params[paramIndex[4]] = angle_value;
	    }
	  } // end if CSC

	  if (writeReport) {
	    report << "reports[-1].phiyFit((" << angle_value << ", " << angle_error << ", " << angle_antisym << "), \\" << std::endl
		   << "                    (" << sigma_value << ", " << sigma_error << ", " << sigma_antisym << "), \\" << std::endl;
	    if (phiyFitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian) {
	    report << "                    (" << gamma_value << ", " << gamma_error << ", " << gamma_antisym << "), \\" << std::endl;
	    }
	    else {
	      report << "                    None, \\" << std::endl;
	    }
	    report << "                    " << redchi2 << ", " << posNum << ", " << negNum << ")" << std::endl;
	  } // end if writeReport
	}
	else if (writeReport) {
	  report << "reports[-1].phiyFit_status = \"FAIL\"" << std::endl;
	}
      }

      if (rphiFitter != m_rphiFitters.end()) {
	// the fit is verbose in std::cout anyway
	std::cout << "=============================================================================================" << std::endl;
	std::cout << "Fitting " << name.str() << " rphi" << std::endl;

	if (phiyOkay  &&  rphiFitter->second->fit(phiyValue)) {
	  std::stringstream name2;
	  name2 << name.str() << "_rphiFit";
	  rphiFitter->second->plot(phiyValue, name2.str(), &rootDirectory);
	  double redchi2 = rphiFitter->second->redchi2(phiyValue, name2.str(), &rootDirectory);
	  long posNum = rphiFitter->second->numResidualsPos();
	  long negNum = rphiFitter->second->numResidualsNeg();

	  double position_value = rphiFitter->second->value(MuonResidualsPositionFitter::kPosition);
	  double position_error = rphiFitter->second->error(MuonResidualsPositionFitter::kPosition);
	  double position_antisym = rphiFitter->second->antisym(MuonResidualsPositionFitter::kPosition);

	  double zpos_value = rphiFitter->second->value(MuonResidualsPositionFitter::kZpos);
	  double zpos_error = rphiFitter->second->error(MuonResidualsPositionFitter::kZpos);
	  double zpos_antisym = rphiFitter->second->antisym(MuonResidualsPositionFitter::kZpos);

	  double phiz_value = rphiFitter->second->value(MuonResidualsPositionFitter::kPhiz);
	  double phiz_error = rphiFitter->second->error(MuonResidualsPositionFitter::kPhiz);
	  double phiz_antisym = rphiFitter->second->antisym(MuonResidualsPositionFitter::kPhiz);

	  double scattering_value = rphiFitter->second->value(MuonResidualsPositionFitter::kScattering);
	  double scattering_error = rphiFitter->second->error(MuonResidualsPositionFitter::kScattering);
	  double scattering_antisym = rphiFitter->second->antisym(MuonResidualsPositionFitter::kScattering);

	  double sigma_value = rphiFitter->second->value(MuonResidualsPositionFitter::kSigma);
	  double sigma_error = rphiFitter->second->error(MuonResidualsPositionFitter::kSigma);
	  double sigma_antisym = rphiFitter->second->antisym(MuonResidualsPositionFitter::kSigma);

	  double gamma_value, gamma_error, gamma_antisym;
	  gamma_value = gamma_error = gamma_antisym = 0.;
	  if (rphiFitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian) {
	    gamma_value = rphiFitter->second->value(MuonResidualsPositionFitter::kGamma);
	    gamma_error = rphiFitter->second->error(MuonResidualsPositionFitter::kGamma);
	    gamma_antisym = rphiFitter->second->antisym(MuonResidualsPositionFitter::kGamma);
	  }

	  if (id.subdetId() == MuonSubdetId::DT) {
	    if (align_x) {
	      params[paramIndex[0]] = position_value;
	      cov[paramIndex[0]][paramIndex[0]] = 0.;   // local x-z is the global x-y plane; with an x alignment, this is now a good parameter
	      cov[paramIndex[2]][paramIndex[2]] = 0.;
	    }

	    if (align_z) {
	      params[paramIndex[2]] = -zpos_value;   // this is the right sign convention
	    }
	  
	    if (align_phiz) {
	      params[paramIndex[5]] = -phiz_value;   // this is the right sign convention
	    }
	  } // end if DT

	  else {
	    if (align_x) {
	      GlobalPoint cscCenter = (*ali)->globalPosition();
	      double R = sqrt(cscCenter.x()*cscCenter.x() + cscCenter.y()*cscCenter.y());
	      double globalphi_correction = position_value / R;
	      
	      double localx_correction = R * sin(globalphi_correction);
	      double localy_correction = R * (cos(globalphi_correction) - 1.);
	      double phiz_correction = -globalphi_correction;

	      params[paramIndex[0]] = localx_correction;
	      params[paramIndex[1]] = localy_correction;
	      params[paramIndex[5]] = phiz_correction;

	      cov[paramIndex[0]][paramIndex[0]] = 0.;  // local x-y plane is the global x-y plane; with an rphi alignment, this is a now a good parameter
	      cov[paramIndex[1]][paramIndex[1]] = 0.;
	    }

	    if (align_z) {
	      params[paramIndex[2]] = -zpos_value;   // this is the right sign convention
	    }

	    if (align_phiz) {
	      // += not =    ...accumulated on top of whatever was needed for curvilinear rphi correction
	      params[paramIndex[5]] -= phiz_value;   // this is the right sign convention
	    }
	  } // end if CSC

	  if (writeReport) {
	    report << "reports[-1].rphiFit((" << position_value << ", " << position_error << ", " << position_antisym << "), \\" << std::endl
		   << "                    (" << zpos_value << ", " << zpos_error << ", " << zpos_antisym << "), \\" << std::endl
		   << "                    (" << phiz_value << ", " << phiz_error << ", " << phiz_antisym << "), \\" << std::endl
		   << "                    (" << scattering_value << ", " << scattering_error << ", " << scattering_antisym << "), \\" << std::endl
		   << "                    (" << sigma_value << ", " << sigma_error << ", " << sigma_antisym << "), \\" << std::endl;
	    if (rphiFitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian) {
	    report << "                    (" << gamma_value << ", " << gamma_error << ", " << gamma_antisym << "), \\" << std::endl;
	    }
	    else {
	      report << "                    None, \\" << std::endl;
	    }
	    report << "                    " << redchi2 << ", " << posNum << ", " << negNum << ")" << std::endl;
	  } // end if writeReport
	}
	else if (writeReport) {
	  report << "reports[-1].rphiFit_status = \"FAIL\"" << std::endl;
	}
      }

      bool phixOkay = false;
      double phixValue = 0.;
      if (phixFitter != m_phixFitters.end()) {
	// the fit is verbose in std::cout anyway
	std::cout << "=============================================================================================" << std::endl;
	std::cout << "Fitting " << name.str() << " phix" << std::endl;

	if (phixFitter->second->fit(0.)) {
	  std::stringstream name2;
	  name2 << name.str() << "_phixFit";
	  phixFitter->second->plot(0., name2.str(), &rootDirectory);
	  double redchi2 = phixFitter->second->redchi2(0., name2.str(), &rootDirectory);
	  long posNum = phixFitter->second->numResidualsPos();
	  long negNum = phixFitter->second->numResidualsNeg();

	  double angle_value = phixFitter->second->value(MuonResidualsAngleFitter::kAngle);
	  double angle_error = phixFitter->second->error(MuonResidualsAngleFitter::kAngle);
	  double angle_antisym = phixFitter->second->antisym(MuonResidualsAngleFitter::kAngle);
	  phixOkay = true;
	  phixValue = angle_value;

	  double sigma_value = phixFitter->second->value(MuonResidualsAngleFitter::kSigma);
	  double sigma_error = phixFitter->second->error(MuonResidualsAngleFitter::kSigma);
	  double sigma_antisym = phixFitter->second->antisym(MuonResidualsAngleFitter::kSigma);

	  double gamma_value, gamma_error, gamma_antisym;
	  gamma_value = gamma_error = gamma_antisym = 0.;
	  if (phixFitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian) {
	    gamma_value = phixFitter->second->value(MuonResidualsAngleFitter::kGamma);
	    gamma_error = phixFitter->second->error(MuonResidualsAngleFitter::kGamma);
	    gamma_antisym = phixFitter->second->antisym(MuonResidualsAngleFitter::kGamma);
	  }

	  if (id.subdetId() == MuonSubdetId::DT) {
	    if (align_phix) {
	      params[paramIndex[3]] = -angle_value;   // confirmed sign
	    }
	  } // end if DT

	  if (writeReport) {
	    report << "reports[-1].phixFit((" << angle_value << ", " << angle_error << ", " << angle_antisym << "), \\" << std::endl
		   << "                    (" << sigma_value << ", " << sigma_error << ", " << sigma_antisym << "), \\" << std::endl;
	    if (phixFitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian) {
	    report << "                    (" << gamma_value << ", " << gamma_error << ", " << gamma_antisym << "), \\" << std::endl;
	    }
	    else {
	      report << "                    None, \\" << std::endl;
	    }
	    report << "                    " << redchi2 << ", " << posNum << ", " << negNum << ")" << std::endl;
	  } // end if writeReport
	}
	else if (writeReport) {
	  report << "reports[-1].phixFit_status = \"FAIL\"" << std::endl;
	}
      }

      if (zFitter != m_zFitters.end()) {
	// the fit is verbose in std::cout anyway
	std::cout << "=============================================================================================" << std::endl;
	std::cout << "Fitting " << name.str() << " z" << std::endl;

	if (phixOkay  &&  zFitter->second->fit(phixValue)) {
	  std::stringstream name2;
	  name2 << name.str() << "_zFit";
	  zFitter->second->plot(phixValue, name2.str(), &rootDirectory);
	  double redchi2 = zFitter->second->redchi2(phixValue, name2.str(), &rootDirectory);
	  long posNum = zFitter->second->numResidualsPos();
	  long negNum = zFitter->second->numResidualsNeg();

	  double position_value = zFitter->second->value(MuonResidualsPositionFitter::kPosition);
	  double position_error = zFitter->second->error(MuonResidualsPositionFitter::kPosition);
	  double position_antisym = zFitter->second->antisym(MuonResidualsPositionFitter::kPosition);

	  double zpos_value = zFitter->second->value(MuonResidualsPositionFitter::kZpos);
	  double zpos_error = zFitter->second->error(MuonResidualsPositionFitter::kZpos);
	  double zpos_antisym = zFitter->second->antisym(MuonResidualsPositionFitter::kZpos);

	  double phiz_value = zFitter->second->value(MuonResidualsPositionFitter::kPhiz);
	  double phiz_error = zFitter->second->error(MuonResidualsPositionFitter::kPhiz);
	  double phiz_antisym = zFitter->second->antisym(MuonResidualsPositionFitter::kPhiz);

	  double scattering_value = zFitter->second->value(MuonResidualsPositionFitter::kScattering);
	  double scattering_error = zFitter->second->error(MuonResidualsPositionFitter::kScattering);
	  double scattering_antisym = zFitter->second->antisym(MuonResidualsPositionFitter::kScattering);

	  double sigma_value = zFitter->second->value(MuonResidualsPositionFitter::kSigma);
	  double sigma_error = zFitter->second->error(MuonResidualsPositionFitter::kSigma);
	  double sigma_antisym = zFitter->second->antisym(MuonResidualsPositionFitter::kSigma);

	  double gamma_value, gamma_error, gamma_antisym;
	  gamma_value = gamma_error = gamma_antisym = 0.;
	  if (zFitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian) {
	    gamma_value = zFitter->second->value(MuonResidualsPositionFitter::kGamma);
	    gamma_error = zFitter->second->error(MuonResidualsPositionFitter::kGamma);
	    gamma_antisym = zFitter->second->antisym(MuonResidualsPositionFitter::kGamma);
	  }

	  if (id.subdetId() == MuonSubdetId::DT) {
	    if (align_x) {
	      params[paramIndex[1]] = position_value;
	      cov[paramIndex[1]][paramIndex[1]] = 0.;   // local y is the global z direction: this is now a good parameter
	    }
	  } // end if DT
	  else { assert(false); } // CSCs don't measure this component: the zFitter should never have been made

	  if (writeReport) {
	    report << "reports[-1].zFit((" << position_value << ", " << position_error << ", " << position_antisym << "), \\" << std::endl
		   << "                 (" << zpos_value << ", " << zpos_error << ", " << zpos_antisym << "), \\" << std::endl
		   << "                 (" << phiz_value << ", " << phiz_error << ", " << phiz_antisym << "), \\" << std::endl
		   << "                 (" << scattering_value << ", " << scattering_error << ", " << scattering_antisym << "), \\" << std::endl
		   << "                 (" << sigma_value << ", " << sigma_error << ", " << sigma_antisym << "), \\" << std::endl;
	    if (zFitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian) {
	    report << "                 (" << gamma_value << ", " << gamma_error << ", " << gamma_antisym << "), \\" << std::endl;
	    }
	    else {
	      report << "                 None, \\" << std::endl;
	    }
	    report << "                 " << redchi2 << ", " << posNum << ", " << negNum << ")" << std::endl;
	  } // end if writeReport
	}
	else if (writeReport) {
	  report << "reports[-1].zFit_status = \"FAIL\"" << std::endl;
	}
      }

      if (writeReport) {
	report << "reports[-1].parameters(";
	if (align_x) report << params[paramIndex[0]] << ", ";
	else report << "None, ";
	if (align_y) report << params[paramIndex[1]] << ", ";
	else report << "None, ";
	if (align_z) report << params[paramIndex[2]] << ", ";
	else report << "None, ";
	if (align_phix) report << params[paramIndex[3]] << ", ";
	else report << "None, ";
	if (align_phiy) report << params[paramIndex[4]] << ", ";
	else report << "None, ";
	if (align_phiz) report << params[paramIndex[5]] << ")" << std::endl;
	else report << "None)" << std::endl;

	report << "reports[-1].errors(";
	if (align_x) report << cov[paramIndex[0]][paramIndex[0]] << ", ";
	else report << "None, ";
	if (align_y) report << cov[paramIndex[1]][paramIndex[1]] << ", ";
	else report << "None, ";
	if (align_z) report << cov[paramIndex[2]][paramIndex[2]] << ")" << std::endl;
	else report << "None)" << std::endl;

	report << std::endl;
      }

      AlignmentParameters *parnew = (*ali)->alignmentParameters()->cloneFromSelected(params, cov);
      (*ali)->setAlignmentParameters(parnew);
      m_alignmentParameterStore->applyParameters(*ali);
      (*ali)->alignmentParameters()->setValid(true);
    } // end loop over alignables

    if (writeReport) report.close();
  }

  // write out the pseudontuples for a later job to collect
  if (m_writeTemporaryFile != std::string("")) {
    FILE *file;
    file = fopen(m_writeTemporaryFile.c_str(), "w");
    int size = m_indexOrder.size();
    fwrite(&size, sizeof(int), 1, file);

    std::vector<unsigned int>::const_iterator index = m_indexOrder.begin();
    std::vector<MuonResidualsTwoBin*>::const_iterator fitter = m_fitterOrder.begin();
    for (int i = 0;  i < size;  ++i, ++index, ++fitter) {
      unsigned int index_towrite = *index;
      fwrite(&index_towrite, sizeof(unsigned int), 1, file);
      (*fitter)->write(file, i);
    }

    fclose(file);
  }
}

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
DEFINE_EDM_PLUGIN(AlignmentAlgorithmPluginFactory, MuonAlignmentFromReference, "MuonAlignmentFromReference");
