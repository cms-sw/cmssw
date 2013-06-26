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
// $Id: MuonAlignmentFromReference.cc,v 1.41 2013/01/07 19:58:00 wmtan Exp $

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/MuonAlignment/interface/AlignableDTSuperLayer.h"
#include "Alignment/MuonAlignment/interface/AlignableDTChamber.h"
#include "Alignment/MuonAlignment/interface/AlignableDTStation.h"
#include "Alignment/MuonAlignment/interface/AlignableDTWheel.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCChamber.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCRing.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCStation.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFromTrack.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResiduals6DOFFitter.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResiduals5DOFFitter.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResiduals6DOFrphiFitter.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsTwoBin.h"

#include "TFile.h"
#include "TTree.h"
#include "TStopwatch.h"

#include <map>
#include <sstream>
#include <fstream>


class MuonAlignmentFromReference : public AlignmentAlgorithmBase
{
public:

  MuonAlignmentFromReference(const edm::ParameterSet& cfg);
  virtual ~MuonAlignmentFromReference();
  
  void initialize(const edm::EventSetup& iSetup,
      AlignableTracker* alignableTracker,
      AlignableMuon* alignableMuon,
      AlignableExtras* extras,
      AlignmentParameterStore* alignmentParameterStore);

  void startNewLoop() {};

  void run(const edm::EventSetup& iSetup, const EventInfo &eventInfo);

  void processMuonResidualsFromTrack(MuonResidualsFromTrack &mrft);

  void terminate(const edm::EventSetup& iSetup);

private:
  bool numeric(std::string s);
  int number(std::string s);
  std::string chamberPrettyNameFromId(unsigned int idx);

  void parseReference(std::vector<Alignable*> &reference, 
      std::vector<Alignable*> &all_DT_chambers, 
      std::vector<Alignable*> &all_CSC_chambers);

  void fitAndAlign();
  void readTmpFiles();
  void writeTmpFiles();

  void selectResidualsPeaks();
  void correctBField();
  void eraseNotSelectedResiduals();

  void fillNtuple();

  // configutarion paramenters:
  edm::InputTag m_muonCollectionTag;
  std::vector<std::string> m_reference;
  double m_minTrackPt;
  double m_maxTrackPt;
  double m_minTrackP;
  double m_maxTrackP;
  double m_maxDxy;
  int m_minTrackerHits;
  double m_maxTrackerRedChi2;
  bool m_allowTIDTEC;
  int m_minNCrossedChambers;
  int m_minDT13Hits;
  int m_minDT2Hits;
  int m_minCSCHits;
  std::string m_writeTemporaryFile;
  std::vector<std::string> m_readTemporaryFiles;
  bool m_doAlignment;
  int m_strategy;
  std::string m_residualsModel;
  int m_minAlignmentHits;
  bool m_twoBin;
  bool m_combineME11;
  bool m_weightAlignment;
  std::string m_reportFileName;
  double m_maxResSlopeY;
  bool m_createNtuple;
  double m_peakNSigma;
  int m_BFieldCorrection;
  bool m_doDT;
  bool m_doCSC;
  std::string m_useResiduals; 
  
  // utility objects
  AlignableNavigator *m_alignableNavigator;
  AlignmentParameterStore *m_alignmentParameterStore;
  std::vector<Alignable*> m_alignables;
  std::map<Alignable*,Alignable*> m_me11map;
  std::map<Alignable*,MuonResidualsTwoBin*> m_fitters;
  std::vector<unsigned int> m_indexes;
  std::map<unsigned int,MuonResidualsTwoBin*> m_fitterOrder;

  // counters
  long m_counter_events;
  long m_counter_tracks;
  long m_counter_trackmomentum;
  long m_counter_trackdxy;
  long m_counter_trackerhits;
  long m_counter_trackerchi2;
  long m_counter_trackertidtec;
  long m_counter_minchambers;
  long m_counter_totchambers;
  long m_counter_station123;
  long m_counter_station123valid;
  long m_counter_station123dt13hits;
  long m_counter_station123dt2hits;
  long m_counter_station123aligning;
  long m_counter_station4;
  long m_counter_station4valid;
  long m_counter_station4hits;
  long m_counter_station4aligning;
  long m_counter_csc;
  long m_counter_cscvalid;
  long m_counter_cschits;
  long m_counter_cscaligning;
  long m_counter_resslopey;

  // debug ntuple
  void bookNtuple();
  TTree * m_ttree;
  MuonResidualsFitter::MuonAlignmentTreeRow m_tree_row;
};


MuonAlignmentFromReference::MuonAlignmentFromReference(const edm::ParameterSet &cfg)
  : AlignmentAlgorithmBase(cfg)
  , m_muonCollectionTag(cfg.getParameter<edm::InputTag>("muonCollectionTag"))
  , m_reference(cfg.getParameter<std::vector<std::string> >("reference"))
  , m_minTrackPt(cfg.getParameter<double>("minTrackPt"))
  , m_maxTrackPt(cfg.getParameter<double>("maxTrackPt"))
  , m_minTrackP(cfg.getParameter<double>("minTrackP"))
  , m_maxTrackP(cfg.getParameter<double>("maxTrackP"))
  , m_maxDxy(cfg.getParameter<double>("maxDxy"))
  , m_minTrackerHits(cfg.getParameter<int>("minTrackerHits"))
  , m_maxTrackerRedChi2(cfg.getParameter<double>("maxTrackerRedChi2"))
  , m_allowTIDTEC(cfg.getParameter<bool>("allowTIDTEC"))
  , m_minNCrossedChambers(cfg.getParameter<int>("minNCrossedChambers"))
  , m_minDT13Hits(cfg.getParameter<int>("minDT13Hits"))
  , m_minDT2Hits(cfg.getParameter<int>("minDT2Hits"))
  , m_minCSCHits(cfg.getParameter<int>("minCSCHits"))
  , m_writeTemporaryFile(cfg.getParameter<std::string>("writeTemporaryFile"))
  , m_readTemporaryFiles(cfg.getParameter<std::vector<std::string> >("readTemporaryFiles"))
  , m_doAlignment(cfg.getParameter<bool>("doAlignment"))
  , m_strategy(cfg.getParameter<int>("strategy"))
  , m_residualsModel(cfg.getParameter<std::string>("residualsModel"))
  , m_minAlignmentHits(cfg.getParameter<int>("minAlignmentHits"))
  , m_twoBin(cfg.getParameter<bool>("twoBin"))
  , m_combineME11(cfg.getParameter<bool>("combineME11"))
  , m_weightAlignment(cfg.getParameter<bool>("weightAlignment"))
  , m_reportFileName(cfg.getParameter<std::string>("reportFileName"))
  , m_maxResSlopeY(cfg.getParameter<double>("maxResSlopeY"))
  , m_createNtuple(cfg.getParameter<bool>("createNtuple"))
  , m_peakNSigma(cfg.getParameter<double>("peakNSigma"))
  , m_BFieldCorrection(cfg.getParameter<int>("bFieldCorrection"))
  , m_doDT(cfg.getParameter<bool>("doDT"))
  , m_doCSC(cfg.getParameter<bool>("doCSC"))
  , m_useResiduals(cfg.getParameter<std::string>("useResiduals"))
{
  // alignment requires a TFile to provide plots to check the fit output
  // just filling the residuals lists does not
  // but we don't want to wait until the end of the job to find out that the TFile is missing
  if (m_doAlignment || m_createNtuple) {
    edm::Service<TFileService> fs;
    TFile &tfile = fs->file();
    tfile.ls();
  }

  m_ttree = NULL;
  if (m_createNtuple) bookNtuple();

  m_counter_events = 0;
  m_counter_tracks = 0;
  m_counter_trackmomentum = 0;
  m_counter_trackdxy = 0;
  m_counter_trackerhits = 0;
  m_counter_trackerchi2 = 0;
  m_counter_trackertidtec = 0;
  m_counter_minchambers = 0;
  m_counter_totchambers = 0;
  m_counter_station123 = 0;
  m_counter_station123valid = 0;
  m_counter_station123dt13hits = 0;
  m_counter_station123dt2hits = 0;
  m_counter_station123aligning = 0;
  m_counter_station4 = 0;
  m_counter_station4valid = 0;
  m_counter_station4hits = 0;
  m_counter_station4aligning = 0;
  m_counter_csc = 0;
  m_counter_cscvalid = 0;
  m_counter_cschits = 0;
  m_counter_cscaligning = 0;
  m_counter_resslopey = 0;
}


MuonAlignmentFromReference::~MuonAlignmentFromReference()
{
  delete m_alignableNavigator;
}


void  MuonAlignmentFromReference::bookNtuple()
{
  edm::Service<TFileService> fs;
  m_ttree = fs->make<TTree>("mual_ttree", "mual_ttree");
  m_ttree->Branch("is_plus", &m_tree_row.is_plus, "is_plus/O");
  m_ttree->Branch("is_dt", &m_tree_row.is_dt, "is_dt/O");
  m_ttree->Branch("station", &m_tree_row.station, "station/b");
  m_ttree->Branch("ring_wheel", &m_tree_row.ring_wheel, "ring_wheel/B");
  m_ttree->Branch("sector", &m_tree_row.sector, "sector/b");
  m_ttree->Branch("res_x", &m_tree_row.res_x, "res_x/F");
  m_ttree->Branch("res_y", &m_tree_row.res_y, "res_y/F");
  m_ttree->Branch("res_slope_x", &m_tree_row.res_slope_x, "res_slope_x/F");
  m_ttree->Branch("res_slope_y", &m_tree_row.res_slope_y, "res_slope_y/F");
  m_ttree->Branch("pos_x",&m_tree_row.pos_x, "pos_x/F");
  m_ttree->Branch("pos_y",&m_tree_row.pos_y, "pos_y/F");
  m_ttree->Branch("angle_x",&m_tree_row.angle_x, "angle_x/F");
  m_ttree->Branch("angle_y",&m_tree_row.angle_y,"angle_y/F");
  m_ttree->Branch("pz",&m_tree_row.pz,"pz/F");
  m_ttree->Branch("pt",&m_tree_row.pt,"pt/F");
  m_ttree->Branch("q",&m_tree_row.q,"q/B");
  m_ttree->Branch("select", &m_tree_row.select, "select/O");
  //m_ttree->Branch("",&m_tree_row.,"/");

}


bool MuonAlignmentFromReference::numeric(std::string s)
{
  return s.length()==1 && std::isdigit(s[0]);
}


int MuonAlignmentFromReference::number(std::string s)
{
  if (!numeric(s)) assert(false);
  return atoi(s.c_str());
}


void MuonAlignmentFromReference::initialize(const edm::EventSetup& iSetup,
    AlignableTracker* alignableTracker,
    AlignableMuon* alignableMuon,
    AlignableExtras* extras,
    AlignmentParameterStore* alignmentParameterStore)
{
   if (alignableMuon == NULL)
     throw cms::Exception("MuonAlignmentFromReference") << "doMuon must be set to True" << std::endl;

   m_alignableNavigator = new AlignableNavigator(alignableMuon);
   m_alignmentParameterStore = alignmentParameterStore;
   m_alignables = m_alignmentParameterStore->alignables();

   int residualsModel;
   if      (m_residualsModel == std::string("pureGaussian"))    residualsModel = MuonResidualsFitter::kPureGaussian;
   else if (m_residualsModel == std::string("pureGaussian2D"))  residualsModel = MuonResidualsFitter::kPureGaussian2D;
   else if (m_residualsModel == std::string("powerLawTails"))   residualsModel = MuonResidualsFitter::kPowerLawTails;
   else if (m_residualsModel == std::string("ROOTVoigt"))       residualsModel = MuonResidualsFitter::kROOTVoigt;
   else if (m_residualsModel == std::string("GaussPowerTails")) residualsModel = MuonResidualsFitter::kGaussPowerTails;
   else throw cms::Exception("MuonAlignmentFromReference") << "unrecognized residualsModel: \"" << m_residualsModel << "\"" << std::endl;

   int useResiduals;
   if      (m_useResiduals == std::string("1111")) useResiduals = MuonResidualsFitter::k1111;
   else if (m_useResiduals == std::string("1110")) useResiduals = MuonResidualsFitter::k1110;
   else if (m_useResiduals == std::string("1100")) useResiduals = MuonResidualsFitter::k1100;
   else if (m_useResiduals == std::string("1010")) useResiduals = MuonResidualsFitter::k1010;
   else if (m_useResiduals == std::string("0010")) useResiduals = MuonResidualsFitter::k0010;
   else throw cms::Exception("MuonAlignmentFromReference") << "unrecognized useResiduals: \"" << m_useResiduals << "\"" << std::endl;

   edm::ESHandle<CSCGeometry> cscGeometry;
   iSetup.get<MuonGeometryRecord>().get(cscGeometry);

   // set up the MuonResidualsFitters (which also collect residuals for fitting)
   m_me11map.clear();
   m_fitters.clear();
   m_indexes.clear();
   m_fitterOrder.clear();

   for (std::vector<Alignable*>::const_iterator ali = m_alignables.begin();  ali != m_alignables.end();  ++ali)
   {
     bool made_fitter = false;

     // fitters for DT
     if ((*ali)->alignableObjectId() == align::AlignableDTChamber)
     {
       DTChamberId id( (*ali)->geomDetId().rawId() );
       
       if (id.station() == 4)
       {
         m_fitters[*ali] =
             new MuonResidualsTwoBin(m_twoBin, new MuonResiduals5DOFFitter(residualsModel, m_minAlignmentHits, useResiduals, m_weightAlignment),
                                               new MuonResiduals5DOFFitter(residualsModel, m_minAlignmentHits, useResiduals, m_weightAlignment));
         made_fitter = true;
       }
       else
       {
         m_fitters[*ali] =
             new MuonResidualsTwoBin(m_twoBin, new MuonResiduals6DOFFitter(residualsModel, m_minAlignmentHits, useResiduals, m_weightAlignment),
                                               new MuonResiduals6DOFFitter(residualsModel, m_minAlignmentHits, useResiduals, m_weightAlignment));
         made_fitter = true;
       }
     }

     // fitters for CSC
     else if ((*ali)->alignableObjectId() == align::AlignableCSCChamber)
     {
       Alignable *thisali = *ali;
       CSCDetId id( (*ali)->geomDetId().rawId() );

       // take care of ME1/1a
       if (m_combineME11  &&  id.station() == 1  &&  id.ring() == 4)
       {
         CSCDetId pairid(id.endcap(), 1, 1, id.chamber());

         for (std::vector<Alignable*>::const_iterator ali2 = m_alignables.begin();  ali2 != m_alignables.end();  ++ali2)
         {
           if ((*ali2)->alignableObjectId() == align::AlignableCSCChamber  &&  (*ali2)->geomDetId().rawId() == pairid.rawId())
           {
             thisali = *ali2;
             break;
           }
         }
         m_me11map[*ali] = thisali;  // points from each ME1/4 chamber to the corresponding ME1/1 chamber
       }

       if (thisali == *ali)   // don't make fitters for ME1/4; they get taken care of in ME1/1
       {
         m_fitters[*ali] =
             new MuonResidualsTwoBin(m_twoBin, new MuonResiduals6DOFrphiFitter(residualsModel, m_minAlignmentHits, useResiduals, &(*cscGeometry), m_weightAlignment),
                                               new MuonResiduals6DOFrphiFitter(residualsModel, m_minAlignmentHits, useResiduals, &(*cscGeometry), m_weightAlignment));
         made_fitter = true;
       }
     }

     else {
       throw cms::Exception("MuonAlignmentFromReference") << "only DTChambers and CSCChambers can be aligned with this module" << std::endl;
     }

     if (made_fitter) {
       m_fitters[*ali]->setStrategy(m_strategy);

       int index = (*ali)->geomDetId().rawId();
       m_indexes.push_back(index);
       m_fitterOrder[index] = m_fitters[*ali];
     }
   } // end loop over chambers chosen for alignment

   // cannonical order of fitters in the file
   std::sort(m_indexes.begin(), m_indexes.end());

   // de-weight all chambers but the reference
   std::vector<Alignable*> all_DT_chambers = alignableMuon->DTChambers();
   std::vector<Alignable*> all_CSC_chambers = alignableMuon->CSCChambers();
   std::vector<Alignable*> reference;
   if (m_reference.size()) parseReference(reference, all_DT_chambers, all_CSC_chambers);
   
   alignmentParameterStore->setAlignmentPositionError(all_DT_chambers, 100000000., 0.);
   alignmentParameterStore->setAlignmentPositionError(all_CSC_chambers, 100000000., 0.);
   alignmentParameterStore->setAlignmentPositionError(reference, 0., 0.);
}


void MuonAlignmentFromReference::parseReference(
    std::vector<Alignable*> &reference, 
    std::vector<Alignable*> &all_DT_chambers, 
    std::vector<Alignable*> &all_CSC_chambers)
{
  std::map<Alignable*,bool> already_seen;

  for (std::vector<std::string>::const_iterator name = m_reference.begin();  name != m_reference.end();  ++name)
  {
    bool parsing_error = false;

    bool barrel = (name->substr(0, 2) == std::string("MB"));
    bool endcap = (name->substr(0, 2) == std::string("ME"));
    if (!barrel  &&  !endcap) parsing_error = true;

    if (!parsing_error  &&  barrel)
    {
      int index = 2;
      if (name->substr(index, 1) == std::string(" "))  index++;

      bool plus = true;
      if (name->substr(index, 1) == std::string("+"))
      {
        plus = true;
        index++;
      }
      else if (name->substr(index, 1) == std::string("-"))
      {
        plus = false;
        index++;
      }
      else if (numeric(name->substr(index, 1))) {}
      else parsing_error = true;

      int wheel = 0;
      bool wheel_digit = false;
      while (!parsing_error  &&  numeric(name->substr(index, 1)))
      {
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
      while (!parsing_error  &&  numeric(name->substr(index, 1)))
      {
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
      while (!parsing_error  &&  numeric(name->substr(index, 1)))
      {
        sector *= 10;
        sector += number(name->substr(index, 1));
        sector_digit = true;
        index++;
      }
      if (!sector_digit) parsing_error = true;

      if (!parsing_error)
      {
        bool no_such_chamber = false;

        if (wheel < -2  ||  wheel > 2) no_such_chamber = true;
        if (station < 1  ||  station > 4) no_such_chamber = true;
        if (station == 4  &&  (sector < 1  ||  sector > 14)) no_such_chamber = true;
        if (station < 4  &&  (sector < 1  ||  sector > 12)) no_such_chamber = true;

        if (no_such_chamber)
          throw cms::Exception("MuonAlignmentFromReference") << "reference chamber doesn't exist: " << (*name) << std::endl;

        DTChamberId id(wheel, station, sector);
        for (std::vector<Alignable*>::const_iterator ali = all_DT_chambers.begin();  ali != all_DT_chambers.end();  ++ali)
        {
          if ((*ali)->geomDetId().rawId() == id.rawId())
          {
            std::map<Alignable*,bool>::const_iterator trial = already_seen.find(*ali);
            if (trial == already_seen.end())
            {
              reference.push_back(*ali);
              already_seen[*ali] = true;
            }
          }
        }
      } // if (!parsing_error)
    }
    
    if (!parsing_error  &&  endcap)
    {
      int index = 2;
      if (name->substr(index, 1) == std::string(" "))  index++;

      bool plus = true;
      if (name->substr(index, 1) == std::string("+"))
      {
        plus = true;
        index++;
      }
      else if (name->substr(index, 1) == std::string("-"))
      {
        plus = false;
        index++;
      }
      else if (numeric(name->substr(index, 1))) {}
      else parsing_error = true;

      int station = 0;
      bool station_digit = false;
      while (!parsing_error  &&  numeric(name->substr(index, 1)))
      {
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
      while (!parsing_error  &&  numeric(name->substr(index, 1)))
      {
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
      while (!parsing_error  &&  numeric(name->substr(index, 1)))
      {
        chamber *= 10;
        chamber += number(name->substr(index, 1));
        chamber_digit = true;
        index++;
      }
      if (!chamber_digit) parsing_error = true;

      if (!parsing_error)
      {
        bool no_such_chamber = false;

        int endcap = (station > 0 ? 1 : 2);
        station = abs(station);
        if (station < 1  ||  station > 4) no_such_chamber = true;
        if (station == 1  &&  (ring < 1  ||  ring > 4)) no_such_chamber = true;
        if (station > 1  &&  (ring < 1  ||  ring > 2)) no_such_chamber = true;
        if (station == 1  &&  (chamber < 1  ||  chamber > 36)) no_such_chamber = true;
        if (station > 1  &&  ring == 1  &&  (chamber < 1  ||  chamber > 18)) no_such_chamber = true;
        if (station > 1  &&  ring == 2  &&  (chamber < 1  ||  chamber > 36)) no_such_chamber = true;

        if (no_such_chamber)
          throw cms::Exception("MuonAlignmentFromReference") << "reference chamber doesn't exist: " << (*name) << std::endl;

        CSCDetId id(endcap, station, ring, chamber);
        for (std::vector<Alignable*>::const_iterator ali = all_CSC_chambers.begin();  ali != all_CSC_chambers.end();  ++ali)
        {
          if ((*ali)->geomDetId().rawId() == id.rawId())
          {
            std::map<Alignable*,bool>::const_iterator trial = already_seen.find(*ali);
            if (trial == already_seen.end()) 
            {
              reference.push_back(*ali);
              already_seen[*ali] = true;
            }
          }
        }
      } // if (!parsing_error)
    }// endcap

    if (parsing_error)
      throw cms::Exception("MuonAlignmentFromReference") << "reference chamber name is malformed: " << (*name) << std::endl;
  }
}


void MuonAlignmentFromReference::run(const edm::EventSetup& iSetup, const EventInfo &eventInfo)
{
  m_counter_events++;

  edm::ESHandle<GlobalTrackingGeometry> globalGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(globalGeometry);

  if (m_muonCollectionTag.label().empty()) // use trajectories
  {
    const ConstTrajTrackPairCollection &trajtracks = eventInfo.trajTrackPairs_;
    for (ConstTrajTrackPairCollection::const_iterator trajtrack = trajtracks.begin();  trajtrack != trajtracks.end();  ++trajtrack)
    {
      m_counter_tracks++;

      const Trajectory* traj = (*trajtrack).first;
      const reco::Track* track = (*trajtrack).second;

      if (m_minTrackPt < track->pt()  &&  track->pt() < m_maxTrackPt && m_minTrackP < track->p()  &&  track->p() < m_maxTrackP)
      {
        m_counter_trackmomentum++;

        if ( fabs(track->dxy(eventInfo.beamSpot_.position())) < m_maxDxy )
        {
          m_counter_trackdxy++;

          MuonResidualsFromTrack muonResidualsFromTrack(globalGeometry, traj, track, m_alignableNavigator, 1000.);

          processMuonResidualsFromTrack(muonResidualsFromTrack);
        }
      } // end if track p is within range
    } // end if track pT is within range
  }
  else // use muons
  {
    /*
    for (reco::MuonCollection::const_iterator muon = eventInfo.muonCollection_->begin();  muon != eventInfo.muonCollection_->end();  ++muon)
    {
      if ( !(muon->isTrackerMuon() && muon->innerTrack().isNonnull() ) ) continue;

      m_counter_tracks++;

      if (m_minTrackPt < muon->pt()  &&  muon->pt() < m_maxTrackPt && m_minTrackP < muon->p()  &&  muon->p() < m_maxTrackP)
      {
        m_counter_trackmomentum++;

        if (fabs(muon->innerTrack()->dxy(eventInfo.beamSpot_.position())) < m_maxDxy)
        {
          m_counter_trackdxy++;

          //std::cout<<"    *** will make MuonResidualsFromTrack ***"<<std::endl;
          MuonResidualsFromTrack muonResidualsFromTrack(globalGeometry, &(*muon), m_alignableNavigator, 100.);
          //std::cout<<"    *** have made MuonResidualsFromTrack ***"<<std::endl;

          //std::cout<<" trk eta="<<muon->eta()<<" ndof="<<muon->innerTrack()->ndof()<<" nchi2="<<muon->innerTrack()->normalizedChi2()
          //         <<" muresnchi2="<<muonResidualsFromTrack.normalizedChi2()<<" muresnhits="<<muonResidualsFromTrack.trackerNumHits()<<std::endl;

          processMuonResidualsFromTrack(muonResidualsFromTrack);
        } // end if track p is within range
      } // end if track pT is within range
    } // end loop over tracks
    */
  }
}


void MuonAlignmentFromReference::processMuonResidualsFromTrack(MuonResidualsFromTrack &mrft)
{
  if (mrft.trackerNumHits() >= m_minTrackerHits)
  {
    m_counter_trackerhits++;
    if (mrft.normalizedChi2() < m_maxTrackerRedChi2)
    {
      m_counter_trackerchi2++;
      if (m_allowTIDTEC  ||  !mrft.contains_TIDTEC())
      {
        m_counter_trackertidtec++;

        std::vector<DetId> chamberIds = mrft.chamberIds();

        if ((int)chamberIds.size() >= m_minNCrossedChambers)
        {
          m_counter_minchambers++;

          char charge = (mrft.getTrack()->charge() > 0 ? 1 : -1);

          for (std::vector<DetId>::const_iterator chamberId = chamberIds.begin();  chamberId != chamberIds.end();  ++chamberId)
          {
            if (chamberId->det() != DetId::Muon) continue;
            m_counter_totchambers++;

            // DT station 1,2,3
            if (m_doDT &&
                chamberId->subdetId() == MuonSubdetId::DT  &&
                DTChamberId(chamberId->rawId()).station() != 4)
            {
              MuonChamberResidual *dt13 = mrft.chamberResidual(*chamberId, MuonChamberResidual::kDT13);
              MuonChamberResidual *dt2 = mrft.chamberResidual(*chamberId, MuonChamberResidual::kDT2);

              m_counter_station123++;
              if (dt13 != NULL  &&  dt2 != NULL)
              {
                m_counter_station123valid++;
                if (dt13->numHits() >= m_minDT13Hits)
                {
                  m_counter_station123dt13hits++;
                  if (dt2->numHits() >= m_minDT2Hits)
                  {
                    m_counter_station123dt2hits++;
                    std::map<Alignable*,MuonResidualsTwoBin*>::const_iterator fitter = m_fitters.find(dt13->chamberAlignable());
                    if (fitter != m_fitters.end())
                    {
                      m_counter_station123aligning++;
                      if (fabs(dt2->resslope()) < m_maxResSlopeY)
                      {
                        m_counter_resslopey++;
                        double *residdata = new double[MuonResiduals6DOFFitter::kNData];
                        residdata[MuonResiduals6DOFFitter::kResidX] = dt13->residual();
                        residdata[MuonResiduals6DOFFitter::kResidY] = dt2->residual();
                        residdata[MuonResiduals6DOFFitter::kResSlopeX] = dt13->resslope();
                        residdata[MuonResiduals6DOFFitter::kResSlopeY] = dt2->resslope();
                        residdata[MuonResiduals6DOFFitter::kPositionX] = dt13->trackx();
                        residdata[MuonResiduals6DOFFitter::kPositionY] = dt13->tracky();
                        residdata[MuonResiduals6DOFFitter::kAngleX] = dt13->trackdxdz();
                        residdata[MuonResiduals6DOFFitter::kAngleY] = dt13->trackdydz();
                        residdata[MuonResiduals6DOFFitter::kRedChi2] = (dt13->chi2() + dt2->chi2()) / double(dt13->ndof() + dt2->ndof());
                        residdata[MuonResiduals6DOFFitter::kPz] = mrft.getTrack()->pz();
                        residdata[MuonResiduals6DOFFitter::kPt] = mrft.getTrack()->pt();
                        residdata[MuonResiduals6DOFFitter::kCharge] = mrft.getTrack()->charge();
                        fitter->second->fill(charge, residdata);
                        // the MuonResidualsFitter will delete the array when it is destroyed
                      }
                    }
                  }
                }
              }
            }

            // DT 4th station
            else if (m_doDT &&
                     chamberId->subdetId() == MuonSubdetId::DT  &&
                     DTChamberId(chamberId->rawId()).station() == 4)
            {
              MuonChamberResidual *dt13 = mrft.chamberResidual(*chamberId, MuonChamberResidual::kDT13);

              m_counter_station4++;
              if (dt13 != NULL)
              {
                m_counter_station4valid++;
                if (dt13->numHits() >= m_minDT13Hits)
                {
                  m_counter_station4hits++;

                  std::map<Alignable*,MuonResidualsTwoBin*>::const_iterator fitter = m_fitters.find(dt13->chamberAlignable());
                  if (fitter != m_fitters.end())
                  {
                    m_counter_station4aligning++;

                    double *residdata = new double[MuonResiduals5DOFFitter::kNData];
                    residdata[MuonResiduals5DOFFitter::kResid] = dt13->residual();
                    residdata[MuonResiduals5DOFFitter::kResSlope] = dt13->resslope();
                    residdata[MuonResiduals5DOFFitter::kPositionX] = dt13->trackx();
                    residdata[MuonResiduals5DOFFitter::kPositionY] = dt13->tracky();
                    residdata[MuonResiduals5DOFFitter::kAngleX] = dt13->trackdxdz();
                    residdata[MuonResiduals5DOFFitter::kAngleY] = dt13->trackdydz();
                    residdata[MuonResiduals5DOFFitter::kRedChi2] = dt13->chi2() / double(dt13->ndof());
                    residdata[MuonResiduals5DOFFitter::kPz] = mrft.getTrack()->pz();
                    residdata[MuonResiduals5DOFFitter::kPt] = mrft.getTrack()->pt();
                    residdata[MuonResiduals5DOFFitter::kCharge] = mrft.getTrack()->charge();		
                    fitter->second->fill(charge, residdata);
                    // the MuonResidualsFitter will delete the array when it is destroyed
                  }
                }
              }
            } // end DT 4th station

            // CSC
            else if (m_doCSC  &&
                     chamberId->subdetId() == MuonSubdetId::CSC)
            {
              MuonChamberResidual *csc = mrft.chamberResidual(*chamberId, MuonChamberResidual::kCSC);
              m_counter_csc++;
              if (csc != NULL)
              {
                m_counter_cscvalid++;
                if (csc->numHits() >= m_minCSCHits)
                {
                  m_counter_cschits++;
                  Alignable *ali = csc->chamberAlignable();

                  CSCDetId id(ali->geomDetId().rawId());
                  if (m_combineME11  &&  id.station() == 1  &&  id.ring() == 4)  ali = m_me11map[ali];

                  std::map<Alignable*,MuonResidualsTwoBin*>::const_iterator fitter = m_fitters.find(ali);
                  if (fitter != m_fitters.end())
                  {
                    m_counter_cscaligning++;
                    double *residdata = new double[MuonResiduals6DOFrphiFitter::kNData];
                    residdata[MuonResiduals6DOFrphiFitter::kResid] = csc->residual();
                    residdata[MuonResiduals6DOFrphiFitter::kResSlope] = csc->resslope();
                    residdata[MuonResiduals6DOFrphiFitter::kPositionX] = csc->trackx();
                    residdata[MuonResiduals6DOFrphiFitter::kPositionY] = csc->tracky();
                    residdata[MuonResiduals6DOFrphiFitter::kAngleX] = csc->trackdxdz();
                    residdata[MuonResiduals6DOFrphiFitter::kAngleY] = csc->trackdydz();
                    residdata[MuonResiduals6DOFrphiFitter::kRedChi2] = csc->chi2() / double(csc->ndof());
                    residdata[MuonResiduals6DOFrphiFitter::kPz] = mrft.getTrack()->pz();
                    residdata[MuonResiduals6DOFrphiFitter::kPt] = mrft.getTrack()->pt();
                    residdata[MuonResiduals6DOFrphiFitter::kCharge] = mrft.getTrack()->charge();
                    fitter->second->fill(charge, residdata);
                    // the MuonResidualsFitter will delete the array when it is destroyed
                  }
                }
              }
            } // end CSC

            else if (m_doDT && m_doCSC) assert(false);

          } // end loop over chamberIds
        } // # crossed muon chambers ok
      } // endcap tracker ok
    } // chi2 ok
  } // trackerNumHits ok
}


void MuonAlignmentFromReference::terminate(const edm::EventSetup& iSetup)
{
  // one-time print-out
  std::cout << "Counters:" << std::endl
	    << "COUNT{ events: " << m_counter_events << " }" << std::endl
	    << "COUNT{  tracks: " << m_counter_tracks << " }" << std::endl
	    << "COUNT{   trackppt: " << m_counter_trackmomentum << " }" << std::endl
	    << "COUNT{    trackdxy: " << m_counter_trackdxy << " }" << std::endl
	    << "COUNT{     trackerhits: " << m_counter_trackerhits << " }" << std::endl
	    << "COUNT{      trackerchi2: " << m_counter_trackerchi2 << " }" << std::endl
	    << "COUNT{       trackertidtec: " << m_counter_trackertidtec << " }" << std::endl
	    << "COUNT{        minnchambers: " << m_counter_minchambers << " }" << std::endl
            << "COUNT{          totchambers: " << m_counter_totchambers << " }" << std::endl
	    << "COUNT{            station123:             " << m_counter_station123 << " }" << std::endl
	    << "COUNT{             station123valid:       " << m_counter_station123valid << " }" << std::endl
	    << "COUNT{              station123dt13hits:   " << m_counter_station123dt13hits << " }" << std::endl
	    << "COUNT{               station123dt2hits:   " << m_counter_station123dt2hits << " }" << std::endl
	    << "COUNT{                station123aligning: " << m_counter_station123aligning << " }" << std::endl
	    << "COUNT{                 resslopey: " << m_counter_resslopey << " }" << std::endl
	    << "COUNT{            station4:            " << m_counter_station4 << " }" << std::endl
	    << "COUNT{             station4valid:      " << m_counter_station4valid << " }" << std::endl
	    << "COUNT{              station4hits:      " << m_counter_station4hits << " }" << std::endl
	    << "COUNT{               station4aligning: " << m_counter_station4aligning << " }" << std::endl
	    << "COUNT{            csc:            " << m_counter_csc << " }" << std::endl
	    << "COUNT{             cscvalid:      " << m_counter_cscvalid << " }" << std::endl
	    << "COUNT{              cschits:      " << m_counter_cschits << " }" << std::endl
	    << "COUNT{               cscaligning: " << m_counter_cscaligning << " }" << std::endl
	    << "That's all!" << std::endl;

  TStopwatch stop_watch;

  // collect temporary files
  if (m_readTemporaryFiles.size() != 0) 
  {
    stop_watch.Start();
    readTmpFiles();
    std::cout <<"readTmpFiles took "<< stop_watch.CpuTime() << " sec" << std::endl;
    stop_watch.Stop();
  }
  
  // select residuals peaks and discard tails if peakNSigma>0 (only while doing alignment)
  if (m_peakNSigma > 0. && m_doAlignment) 
  {
    stop_watch.Start();
    selectResidualsPeaks();
    std::cout <<"selectResidualsPeaks took "<< stop_watch.CpuTime() << " sec" << std::endl;
    stop_watch.Stop();
  }

  if (m_BFieldCorrection > 0 && m_doAlignment)
  {
    stop_watch.Start();
    correctBField();
    std::cout <<"correctBField took "<< stop_watch.CpuTime() << " sec" << std::endl;
    stop_watch.Stop();
  }

  // optionally, create an nutuple for easy debugging
  if (m_createNtuple)
  {
    stop_watch.Start();
    fillNtuple();
    std::cout <<"fillNtuple took "<< stop_watch.CpuTime() << " sec" << std::endl;
    stop_watch.Stop();
  }

  if (m_doAlignment)
  {
    stop_watch.Start();
    eraseNotSelectedResiduals();
    std::cout <<"eraseNotSelectedResiduals took "<< stop_watch.CpuTime() << " sec" << std::endl;
    stop_watch.Stop();
  }

  // fit and align (time-consuming, so the user can turn it off if in a residuals-gathering job)
  if (m_doAlignment) 
  {
    stop_watch.Start();
    fitAndAlign();
    std::cout <<"fitAndAlign took "<< stop_watch.CpuTime() << " sec" << std::endl;
    stop_watch.Stop();
  }

  // write out the pseudontuples for a later job to collect
  if (m_writeTemporaryFile != std::string("")) writeTmpFiles();
}


void MuonAlignmentFromReference::fitAndAlign()
{
  edm::Service<TFileService> tfileService;
  TFileDirectory rootDirectory(tfileService->mkdir("MuonAlignmentFromReference"));

  std::ofstream report;
  bool writeReport = (m_reportFileName != std::string(""));
  if (writeReport)
  {
    report.open(m_reportFileName.c_str());
    report << "nan = None;  NAN = None" << std::endl;
    report << "reports = []" << std::endl;
    report << "class ValErr:" << std::endl
           << "    def __init__(self, value, error, antisym):" << std::endl
           << "        self.value, self.error, self.antisym = value, error, antisym" << std::endl
           << "" << std::endl
           << "    def __repr__(self):" << std::endl
           << "        if self.antisym == 0.:" << std::endl
           << "            return \"%g +- %g\" % (self.value, self.error)" << std::endl
           << "        else:" << std::endl
           << "            return \"%g +- %g ~ %g\" % (self.value, self.error, self.antisym)" << std::endl
           << "" << std::endl
           << "class Report:" << std::endl
           << "    def __init__(self, chamberId, postal_address, name):" << std::endl
           << "        self.chamberId, self.postal_address, self.name = chamberId, postal_address, name" << std::endl
           << "        self.status = \"NOFIT\"" << std::endl
           << "        self.fittype = None" << std::endl
           << "" << std::endl
           << "    def add_parameters(self, deltax, deltay, deltaz, deltaphix, deltaphiy, deltaphiz, loglikelihood, numsegments, sumofweights, redchi2):" << std::endl
           << "        self.status = \"PASS\"" << std::endl
           << "        self.deltax, self.deltay, self.deltaz, self.deltaphix, self.deltaphiy, self.deltaphiz = deltax, deltay, deltaz, deltaphix, deltaphiy, deltaphiz" << std::endl
           << "        self.loglikelihood, self.numsegments, self.sumofweights, self.redchi2 = loglikelihood, numsegments, sumofweights, redchi2" << std::endl
           << "" << std::endl
           << "    def add_stats(self, median_x, median_y, median_dxdz, median_dydz, mean30_x, mean30_y, mean20_dxdz, mean50_dydz, mean15_x, mean15_y, mean10_dxdz, mean25_dydz, wmean30_x, wmean30_y, wmean20_dxdz, wmean50_dydz, wmean15_x, wmean15_y, wmean10_dxdz, wmean25_dydz, stdev30_x, stdev30_y, stdev20_dxdz, stdev50_dydz, stdev15_x, stdev15_y, stdev10_dxdz, stdev25_dydz):" << std::endl
           << "        self.median_x, self.median_y, self.median_dxdz, self.median_dydz, self.mean30_x, self.mean30_y, self.mean20_dxdz, self.mean50_dydz, self.mean15_x, self.mean15_y, self.mean10_dxdz, self.mean25_dydz, self.wmean30_x, self.wmean30_y, self.wmean20_dxdz, self.wmean50_dydz, self.wmean15_x, self.wmean15_y, self.wmean10_dxdz, self.wmean25_dydz, self.stdev30_x, self.stdev30_y, self.stdev20_dxdz, self.stdev50_dydz, self.stdev15_x, self.stdev15_y, self.stdev10_dxdz, self.stdev25_dydz = median_x, median_y, median_dxdz, median_dydz, mean30_x, mean30_y, mean20_dxdz, mean50_dydz, mean15_x, mean15_y, mean10_dxdz, mean25_dydz, wmean30_x, wmean30_y, wmean20_dxdz, wmean50_dydz, wmean15_x, wmean15_y, wmean10_dxdz, wmean25_dydz, stdev30_x, stdev30_y, stdev20_dxdz, stdev50_dydz, stdev15_x, stdev15_y, stdev10_dxdz, stdev25_dydz" << std::endl
           << "" << std::endl
           << "    def __repr__(self):" << std::endl
           << "        return \"<Report %s %s %s>\" % (self.postal_address[0], \" \".join(map(str, self.postal_address[1:])), self.status)"<< std::endl
           << std::endl;
  }

  for (std::vector<Alignable*>::const_iterator ali = m_alignables.begin(); ali != m_alignables.end(); ++ali)
  {
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

    DetId id = (*ali)->geomDetId();

    Alignable *thisali = *ali;
    if (m_combineME11 && id.subdetId() == MuonSubdetId::CSC)
    {
      CSCDetId cscid(id.rawId());
      if (cscid.station() == 1 && cscid.ring() == 4)   thisali = m_me11map[*ali];
    }

    char cname[40];
    char wheel_label[][2]={"A","B","C","D","E"};
    
    if (id.subdetId() == MuonSubdetId::DT)
    {
      DTChamberId chamberId(id.rawId());

      //if ( ! ( (chamberId.station()==1&&chamberId.wheel()==0) || (chamberId.station()==4&&chamberId.wheel()==2) ) ) continue;
      
      sprintf(cname, "MBwh%sst%dsec%02d", wheel_label[chamberId.wheel()+2], chamberId.station(), chamberId.sector());
      if (writeReport)
      {
        report << "reports.append(Report(" << id.rawId() << ", (\"DT\", "
            << chamberId.wheel() << ", " << chamberId.station() << ", " << chamberId.sector() << "), \"" << cname << "\"))" << std::endl;
      }
    }
    else if (id.subdetId() == MuonSubdetId::CSC)
    {
      CSCDetId chamberId(id.rawId());
      sprintf(cname, "ME%s%d%d_%02d", (chamberId.endcap() == 1 ? "p" : "m"), chamberId.station(), chamberId.ring(), chamberId.chamber());

      //if ( chamberId.chamber()>6 || chamberId.endcap()==2 || ! ( (chamberId.station()==2&&chamberId.ring()==1) || (chamberId.station()==3&&chamberId.ring()==2) ) ) continue;

      if (writeReport)
      {
        report << "reports.append(Report(" << id.rawId() << ", (\"CSC\", "
            << chamberId.endcap() << ", " << chamberId.station() << ", " << chamberId.ring() << ", " << chamberId.chamber()
            << "), \"" << cname << "\"))" << std::endl;
      }
    }

    //if(! ( strcmp(cname,"MBwhCst3sec12")==0 || strcmp(cname,"MBwhCst3sec06")==0)) continue;

    std::map<Alignable*, MuonResidualsTwoBin*>::const_iterator fitter = m_fitters.find(thisali);

    if (fitter != m_fitters.end())
    {
      //if (fitter->second->type() != MuonResidualsFitter::k6DOFrphi) continue;
      
      TStopwatch stop_watch;
      stop_watch.Start();

      // MINUIT is verbose in std::cout anyway
      std::cout << "=============================================================================================" << std::endl;
      std::cout << "Fitting " << cname << std::endl;

      if (writeReport)
      {
        report << "reports[-1].posNum = " << fitter->second->numResidualsPos() << std::endl;
        report << "reports[-1].negNum = " << fitter->second->numResidualsNeg() << std::endl;
      }

      if (fitter->second->type() == MuonResidualsFitter::k5DOF)
      {
        if (!align_x) fitter->second->fix(MuonResiduals5DOFFitter::kAlignX);
        if (!align_z) fitter->second->fix(MuonResiduals5DOFFitter::kAlignZ);
        if (!align_phix) fitter->second->fix(MuonResiduals5DOFFitter::kAlignPhiX);
        if (!align_phiy) fitter->second->fix(MuonResiduals5DOFFitter::kAlignPhiY);
        if (!align_phiz) fitter->second->fix(MuonResiduals5DOFFitter::kAlignPhiZ);
      }
      else if (fitter->second->type() == MuonResidualsFitter::k6DOF)
      {
        if (!align_x) fitter->second->fix(MuonResiduals6DOFFitter::kAlignX);
        if (!align_y) fitter->second->fix(MuonResiduals6DOFFitter::kAlignY);
        if (!align_z) fitter->second->fix(MuonResiduals6DOFFitter::kAlignZ);
        if (!align_phix) fitter->second->fix(MuonResiduals6DOFFitter::kAlignPhiX);
        if (!align_phiy) fitter->second->fix(MuonResiduals6DOFFitter::kAlignPhiY);
        if (!align_phiz) fitter->second->fix(MuonResiduals6DOFFitter::kAlignPhiZ);
      }
      else if (fitter->second->type() == MuonResidualsFitter::k6DOFrphi)
      {
        if (!align_x) fitter->second->fix(MuonResiduals6DOFrphiFitter::kAlignX);
        if (!align_y) fitter->second->fix(MuonResiduals6DOFrphiFitter::kAlignY);
        if (!align_z) fitter->second->fix(MuonResiduals6DOFrphiFitter::kAlignZ);
        if (!align_phix) fitter->second->fix(MuonResiduals6DOFrphiFitter::kAlignPhiX);
        if (!align_phiy) fitter->second->fix(MuonResiduals6DOFrphiFitter::kAlignPhiY);
        if (!align_phiz) fitter->second->fix(MuonResiduals6DOFrphiFitter::kAlignPhiZ);
      }
      else  assert(false);

      AlgebraicVector params(numParams);
      AlgebraicSymMatrix cov(numParams);

      if (fitter->second->numsegments() >= m_minAlignmentHits)
      {
        bool successful_fit = fitter->second->fit(thisali);

        double loglikelihood = fitter->second->loglikelihood();
        double numsegments = fitter->second->numsegments();
        double sumofweights = fitter->second->sumofweights();
        double redchi2 = fitter->second->plot(cname, &rootDirectory, thisali);

        if (fitter->second->type() == MuonResidualsFitter::k5DOF)
        {
          double deltax_value = fitter->second->value(MuonResiduals5DOFFitter::kAlignX);
          double deltax_error = fitter->second->errorerror(MuonResiduals5DOFFitter::kAlignX);
          double deltax_antisym = fitter->second->antisym(MuonResiduals5DOFFitter::kAlignX);

          double deltaz_value = fitter->second->value(MuonResiduals5DOFFitter::kAlignZ);
          double deltaz_error = fitter->second->errorerror(MuonResiduals5DOFFitter::kAlignZ);
          double deltaz_antisym = fitter->second->antisym(MuonResiduals5DOFFitter::kAlignZ);

          double deltaphix_value = fitter->second->value(MuonResiduals5DOFFitter::kAlignPhiX);
          double deltaphix_error = fitter->second->errorerror(MuonResiduals5DOFFitter::kAlignPhiX);
          double deltaphix_antisym = fitter->second->antisym(MuonResiduals5DOFFitter::kAlignPhiX);

          double deltaphiy_value = fitter->second->value(MuonResiduals5DOFFitter::kAlignPhiY);
          double deltaphiy_error = fitter->second->errorerror(MuonResiduals5DOFFitter::kAlignPhiY);
          double deltaphiy_antisym = fitter->second->antisym(MuonResiduals5DOFFitter::kAlignPhiY);

          double deltaphiz_value = fitter->second->value(MuonResiduals5DOFFitter::kAlignPhiZ);
          double deltaphiz_error = fitter->second->errorerror(MuonResiduals5DOFFitter::kAlignPhiZ);
          double deltaphiz_antisym = fitter->second->antisym(MuonResiduals5DOFFitter::kAlignPhiZ);

          double sigmaresid_value = fitter->second->value(MuonResiduals5DOFFitter::kResidSigma);
          double sigmaresid_error = fitter->second->errorerror(MuonResiduals5DOFFitter::kResidSigma);
          double sigmaresid_antisym = fitter->second->antisym(MuonResiduals5DOFFitter::kResidSigma);

          double sigmaresslope_value = fitter->second->value(MuonResiduals5DOFFitter::kResSlopeSigma);
          double sigmaresslope_error = fitter->second->errorerror(MuonResiduals5DOFFitter::kResSlopeSigma);
          double sigmaresslope_antisym = fitter->second->antisym(MuonResiduals5DOFFitter::kResSlopeSigma);

          double gammaresid_value, gammaresid_error, gammaresid_antisym, gammaresslope_value, gammaresslope_error, gammaresslope_antisym;
          gammaresid_value = gammaresid_error = gammaresid_antisym = gammaresslope_value = gammaresslope_error = gammaresslope_antisym = 0.;

          if (fitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian &&
              fitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian2D &&
              fitter->second->residualsModel() != MuonResidualsFitter::kGaussPowerTails)
          {
            gammaresid_value = fitter->second->value(MuonResiduals5DOFFitter::kResidGamma);
            gammaresid_error = fitter->second->errorerror(MuonResiduals5DOFFitter::kResidGamma);
            gammaresid_antisym = fitter->second->antisym(MuonResiduals5DOFFitter::kResidGamma);

            gammaresslope_value = fitter->second->value(MuonResiduals5DOFFitter::kResSlopeGamma);
            gammaresslope_error = fitter->second->errorerror(MuonResiduals5DOFFitter::kResSlopeGamma);
            gammaresslope_antisym = fitter->second->antisym(MuonResiduals5DOFFitter::kResSlopeGamma);
          }

          if (writeReport)
          {
            report << "reports[-1].fittype = \"5DOF\"" << std::endl;
            report << "reports[-1].add_parameters(ValErr(" << deltax_value << ", " << deltax_error << ", "  << deltax_antisym << "), \\" << std::endl
                   << "                           None, \\" << std::endl
                   << "                           ValErr(" << deltaz_value << ", " << deltaz_error << ", " << deltaz_antisym << "), \\" << std::endl
                   << "                           ValErr(" << deltaphix_value << ", " << deltaphix_error << ", " << deltaphix_antisym << "), \\" << std::endl
                   << "                           ValErr(" << deltaphiy_value << ", " << deltaphiy_error << ", " << deltaphiy_antisym << "), \\" << std::endl
                   << "                           ValErr(" << deltaphiz_value << ", " << deltaphiz_error << ", " << deltaphiz_antisym << "), \\" << std::endl
                   << "                           " << loglikelihood << ", " << numsegments << ", " << sumofweights << ", " << redchi2 << ")" << std::endl;
            report << "reports[-1].sigmaresid = ValErr(" << sigmaresid_value << ", " << sigmaresid_error << ", "  << sigmaresid_antisym << ")" << std::endl;
            report << "reports[-1].sigmaresslope = ValErr(" << sigmaresslope_value << ", " << sigmaresslope_error << ", " << sigmaresslope_antisym << ")" << std::endl;
            if (fitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian &&
                fitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian2D &&
                fitter->second->residualsModel() != MuonResidualsFitter::kGaussPowerTails)
            {
              report << "reports[-1].gammaresid = ValErr(" << gammaresid_value << ", " << gammaresid_error << ", " << gammaresid_antisym << ")" << std::endl;
              report << "reports[-1].gammaresslope = ValErr(" << gammaresslope_value << ", " << gammaresslope_error << ", " << gammaresslope_antisym << ")" << std::endl;
            }

            report << "reports[-1].add_stats(" << fitter->second->median(MuonResiduals5DOFFitter::kResid) << ", " << "None, "
                << fitter->second->median(MuonResiduals5DOFFitter::kResSlope) << ", " << "None, "
                << fitter->second->mean(MuonResiduals5DOFFitter::kResid, 30.) << ", " << "None, "
                << fitter->second->mean(MuonResiduals5DOFFitter::kResSlope, 20.) << ", " << "None, "
                << fitter->second->mean(MuonResiduals5DOFFitter::kResid, 15.) << ", " << "None, "
                << fitter->second->mean(MuonResiduals5DOFFitter::kResSlope, 10.) << ", " << "None, "
                << fitter->second->wmean(MuonResiduals5DOFFitter::kResid, MuonResiduals5DOFFitter::kRedChi2, 30.) << ", " << "None, "
                << fitter->second->wmean(MuonResiduals5DOFFitter::kResSlope, MuonResiduals5DOFFitter::kRedChi2, 20.) << ", " << "None, "
                << fitter->second->wmean(MuonResiduals5DOFFitter::kResid, MuonResiduals5DOFFitter::kRedChi2, 15.) << ", " << "None, "
                << fitter->second->wmean(MuonResiduals5DOFFitter::kResSlope, MuonResiduals5DOFFitter::kRedChi2, 10.) << ", " << "None, "
                << fitter->second->stdev(MuonResiduals5DOFFitter::kResid, 30.) << ", " << "None, "
                << fitter->second->stdev(MuonResiduals5DOFFitter::kResSlope, 20.) << ", " << "None, "
                << fitter->second->stdev(MuonResiduals5DOFFitter::kResid, 15.) << ", " << "None, "
                << fitter->second->stdev(MuonResiduals5DOFFitter::kResSlope, 10.) << ", " << "None)" << std::endl;

            std::stringstream namesimple_x, namesimple_dxdz, nameweighted_x, nameweighted_dxdz;
            namesimple_x << cname << "_simple_x";
            namesimple_dxdz << cname << "_simple_dxdz";
            nameweighted_x << cname << "_weighted_x";
            nameweighted_dxdz << cname << "_weighted_dxdz";

            fitter->second->plotsimple(namesimple_x.str(), &rootDirectory, MuonResiduals5DOFFitter::kResid, 10.);
            fitter->second->plotsimple(namesimple_dxdz.str(), &rootDirectory, MuonResiduals5DOFFitter::kResSlope, 1000.);

            fitter->second->plotweighted(nameweighted_x.str(), &rootDirectory, MuonResiduals5DOFFitter::kResid, MuonResiduals5DOFFitter::kRedChi2, 10.);
            fitter->second->plotweighted(nameweighted_dxdz.str(), &rootDirectory, MuonResiduals5DOFFitter::kResSlope, MuonResiduals5DOFFitter::kRedChi2, 1000.);
          }

          if (successful_fit)
          {
            if (align_x) params[paramIndex[0]] = deltax_value;
            if (align_z) params[paramIndex[2]] = deltaz_value;
            if (align_phix) params[paramIndex[3]] = deltaphix_value;
            if (align_phiy) params[paramIndex[4]] = deltaphiy_value;
            if (align_phiz) params[paramIndex[5]] = deltaphiz_value;
          }
        } // end if 5DOF

        else if (fitter->second->type() == MuonResidualsFitter::k6DOF)
        {
          double deltax_value = fitter->second->value(MuonResiduals6DOFFitter::kAlignX);
          double deltax_error = fitter->second->errorerror(MuonResiduals6DOFFitter::kAlignX);
          double deltax_antisym = fitter->second->antisym(MuonResiduals6DOFFitter::kAlignX);

          double deltay_value = fitter->second->value(MuonResiduals6DOFFitter::kAlignY);
          double deltay_error = fitter->second->errorerror(MuonResiduals6DOFFitter::kAlignY);
          double deltay_antisym = fitter->second->antisym(MuonResiduals6DOFFitter::kAlignY);

          double deltaz_value = fitter->second->value(MuonResiduals6DOFFitter::kAlignZ);
          double deltaz_error = fitter->second->errorerror(MuonResiduals6DOFFitter::kAlignZ);
          double deltaz_antisym = fitter->second->antisym(MuonResiduals6DOFFitter::kAlignZ);

          double deltaphix_value = fitter->second->value(MuonResiduals6DOFFitter::kAlignPhiX);
          double deltaphix_error = fitter->second->errorerror(MuonResiduals6DOFFitter::kAlignPhiX);
          double deltaphix_antisym = fitter->second->antisym(MuonResiduals6DOFFitter::kAlignPhiX);

          double deltaphiy_value = fitter->second->value(MuonResiduals6DOFFitter::kAlignPhiY);
          double deltaphiy_error = fitter->second->errorerror(MuonResiduals6DOFFitter::kAlignPhiY);
          double deltaphiy_antisym = fitter->second->antisym(MuonResiduals6DOFFitter::kAlignPhiY);

          double deltaphiz_value = fitter->second->value(MuonResiduals6DOFFitter::kAlignPhiZ);
          double deltaphiz_error = fitter->second->errorerror(MuonResiduals6DOFFitter::kAlignPhiZ);
          double deltaphiz_antisym = fitter->second->antisym(MuonResiduals6DOFFitter::kAlignPhiZ);

          double sigmax_value = fitter->second->value(MuonResiduals6DOFFitter::kResidXSigma);
          double sigmax_error = fitter->second->errorerror(MuonResiduals6DOFFitter::kResidXSigma);
          double sigmax_antisym = fitter->second->antisym(MuonResiduals6DOFFitter::kResidXSigma);

          double sigmay_value = fitter->second->value(MuonResiduals6DOFFitter::kResidYSigma);
          double sigmay_error = fitter->second->errorerror(MuonResiduals6DOFFitter::kResidYSigma);
          double sigmay_antisym = fitter->second->antisym(MuonResiduals6DOFFitter::kResidYSigma);

          double sigmadxdz_value = fitter->second->value(MuonResiduals6DOFFitter::kResSlopeXSigma);
          double sigmadxdz_error = fitter->second->errorerror(MuonResiduals6DOFFitter::kResSlopeXSigma);
          double sigmadxdz_antisym = fitter->second->antisym(MuonResiduals6DOFFitter::kResSlopeXSigma);

          double sigmadydz_value = fitter->second->value(MuonResiduals6DOFFitter::kResSlopeYSigma);
          double sigmadydz_error = fitter->second->errorerror(MuonResiduals6DOFFitter::kResSlopeYSigma);
          double sigmadydz_antisym = fitter->second->antisym(MuonResiduals6DOFFitter::kResSlopeYSigma);

          double gammax_value, gammax_error, gammax_antisym, gammay_value, gammay_error, gammay_antisym,
              gammadxdz_value, gammadxdz_error, gammadxdz_antisym, gammadydz_value, gammadydz_error, gammadydz_antisym;
          gammax_value = gammax_error = gammax_antisym = gammay_value = gammay_error = gammay_antisym = gammadxdz_value
              = gammadxdz_error = gammadxdz_antisym = gammadydz_value = gammadydz_error = gammadydz_antisym = 0.;
          if (fitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian &&
              fitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian2D &&
              fitter->second->residualsModel() != MuonResidualsFitter::kGaussPowerTails)
          {
            gammax_value = fitter->second->value(MuonResiduals6DOFFitter::kResidXGamma);
            gammax_error = fitter->second->errorerror(MuonResiduals6DOFFitter::kResidXGamma);
            gammax_antisym = fitter->second->antisym(MuonResiduals6DOFFitter::kResidXGamma);

            gammay_value = fitter->second->value(MuonResiduals6DOFFitter::kResidYGamma);
            gammay_error = fitter->second->errorerror(MuonResiduals6DOFFitter::kResidYGamma);
            gammay_antisym = fitter->second->antisym(MuonResiduals6DOFFitter::kResidYGamma);

            gammadxdz_value = fitter->second->value(MuonResiduals6DOFFitter::kResSlopeXGamma);
            gammadxdz_error = fitter->second->errorerror(MuonResiduals6DOFFitter::kResSlopeXGamma);
            gammadxdz_antisym = fitter->second->antisym(MuonResiduals6DOFFitter::kResSlopeXGamma);

            gammadydz_value = fitter->second->value(MuonResiduals6DOFFitter::kResSlopeYGamma);
            gammadydz_error = fitter->second->errorerror(MuonResiduals6DOFFitter::kResSlopeYGamma);
            gammadydz_antisym = fitter->second->antisym(MuonResiduals6DOFFitter::kResSlopeYGamma);
          }

          if (writeReport)
          {
            report << "reports[-1].fittype = \"6DOF\"" << std::endl;
            report << "reports[-1].add_parameters(ValErr(" << deltax_value << ", " << deltax_error << ", " << deltax_antisym << "), \\" << std::endl
                << "                           ValErr(" << deltay_value << ", " << deltay_error << ", " << deltay_antisym << "), \\" << std::endl
                << "                           ValErr(" << deltaz_value << ", " << deltaz_error << ", " << deltaz_antisym << "), \\" << std::endl
                << "                           ValErr(" << deltaphix_value << ", " << deltaphix_error << ", " << deltaphix_antisym << "), \\" << std::endl
                << "                           ValErr(" << deltaphiy_value << ", " << deltaphiy_error << ", " << deltaphiy_antisym << "), \\" << std::endl
                << "                           ValErr(" << deltaphiz_value << ", " << deltaphiz_error << ", " << deltaphiz_antisym << "), \\" << std::endl
                << "                           " << loglikelihood << ", " << numsegments << ", " << sumofweights << ", " << redchi2 << ")" << std::endl;
            report << "reports[-1].sigmax = ValErr(" << sigmax_value << ", " << sigmax_error << ", " << sigmax_antisym<< ")" << std::endl;
            report << "reports[-1].sigmay = ValErr(" << sigmay_value << ", " << sigmay_error << ", " << sigmay_antisym<< ")" << std::endl;
            report << "reports[-1].sigmadxdz = ValErr(" << sigmadxdz_value << ", " << sigmadxdz_error << ", "<< sigmadxdz_antisym << ")" << std::endl;
            report << "reports[-1].sigmadydz = ValErr(" << sigmadydz_value << ", " << sigmadydz_error << ", "<< sigmadydz_antisym << ")" << std::endl;
            if (fitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian &&
                fitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian2D &&
                fitter->second->residualsModel() != MuonResidualsFitter::kGaussPowerTails)
            {
              report << "reports[-1].gammax = ValErr(" << gammax_value << ", " << gammax_error << ", " << gammax_antisym << ")" << std::endl;
              report << "reports[-1].gammay = ValErr(" << gammay_value << ", " << gammay_error << ", " << gammay_antisym << ")" << std::endl;
              report << "reports[-1].gammadxdz = ValErr(" << gammadxdz_value << ", " << gammadxdz_error << ", " << gammadxdz_antisym << ")" << std::endl;
              report << "reports[-1].gammadydz = ValErr(" << gammadydz_value << ", " << gammadydz_error << ", " << gammadydz_antisym << ")" << std::endl;
            }

            report << "reports[-1].add_stats("
                << fitter->second->median(MuonResiduals6DOFFitter::kResidX) << ", "
                << fitter->second->median(MuonResiduals6DOFFitter::kResidY) << ", "
                << fitter->second->median(MuonResiduals6DOFFitter::kResSlopeX) << ", "
                << fitter->second->median(MuonResiduals6DOFFitter::kResSlopeY) << ", "
                << fitter->second->mean(MuonResiduals6DOFFitter::kResidX, 30.) << ", "
                << fitter->second->mean(MuonResiduals6DOFFitter::kResidY, 30.) << ", "
                << fitter->second->mean(MuonResiduals6DOFFitter::kResSlopeX, 20.) << ", "
                << fitter->second->mean(MuonResiduals6DOFFitter::kResSlopeY, 50.) << ", "
                << fitter->second->mean(MuonResiduals6DOFFitter::kResidX, 15.) << ", "
                << fitter->second->mean(MuonResiduals6DOFFitter::kResidY, 15.) << ", "
                << fitter->second->mean(MuonResiduals6DOFFitter::kResSlopeX, 10.) << ", "
                << fitter->second->mean(MuonResiduals6DOFFitter::kResSlopeY, 25.) << ", "
                << fitter->second->wmean(MuonResiduals6DOFFitter::kResidX, MuonResiduals6DOFFitter::kRedChi2, 30.) << ", "
                << fitter->second->wmean(MuonResiduals6DOFFitter::kResidY, MuonResiduals6DOFFitter::kRedChi2, 30.) << ", "
                << fitter->second->wmean(MuonResiduals6DOFFitter::kResSlopeX,MuonResiduals6DOFFitter::kRedChi2, 20.) << ", "
                << fitter->second->wmean(MuonResiduals6DOFFitter::kResSlopeY, MuonResiduals6DOFFitter::kRedChi2, 50.) << ", "
                << fitter->second->wmean(MuonResiduals6DOFFitter::kResidX, MuonResiduals6DOFFitter::kRedChi2, 15.) << ", "
                << fitter->second->wmean(MuonResiduals6DOFFitter::kResidY, MuonResiduals6DOFFitter::kRedChi2, 15.) << ", "
                << fitter->second->wmean(MuonResiduals6DOFFitter::kResSlopeX, MuonResiduals6DOFFitter::kRedChi2, 10.) << ", "
                << fitter->second->wmean(MuonResiduals6DOFFitter::kResSlopeY, MuonResiduals6DOFFitter::kRedChi2, 25.) << ", "
                << fitter->second->stdev(MuonResiduals6DOFFitter::kResidX, 30.) << ", "
                << fitter->second->stdev(MuonResiduals6DOFFitter::kResidY, 30.) << ", "
                << fitter->second->stdev(MuonResiduals6DOFFitter::kResSlopeX, 20.) << ", "
                << fitter->second->stdev(MuonResiduals6DOFFitter::kResSlopeY, 50.) << ", "
                << fitter->second->stdev(MuonResiduals6DOFFitter::kResidX, 15.) << ", "
                << fitter->second->stdev(MuonResiduals6DOFFitter::kResidY, 15.) << ", "
                << fitter->second->stdev(MuonResiduals6DOFFitter::kResSlopeX, 10.) << ", "
                << fitter->second->stdev(MuonResiduals6DOFFitter::kResSlopeY, 25.) << ")" << std::endl;

            std::stringstream namesimple_x, namesimple_y, namesimple_dxdz, namesimple_dydz, nameweighted_x,
                nameweighted_y, nameweighted_dxdz, nameweighted_dydz;
            namesimple_x << cname << "_simple_x";
            namesimple_y << cname << "_simple_y";
            namesimple_dxdz << cname << "_simple_dxdz";
            namesimple_dydz << cname << "_simple_dydz";
            nameweighted_x << cname << "_weighted_x";
            nameweighted_y << cname << "_weighted_y";
            nameweighted_dxdz << cname << "_weighted_dxdz";
            nameweighted_dydz << cname << "_weighted_dydz";

            fitter->second->plotsimple(namesimple_x.str(), &rootDirectory, MuonResiduals6DOFFitter::kResidX, 10.);
            fitter->second->plotsimple(namesimple_y.str(), &rootDirectory, MuonResiduals6DOFFitter::kResidY, 10.);
            fitter->second->plotsimple(namesimple_dxdz.str(), &rootDirectory, MuonResiduals6DOFFitter::kResSlopeX, 1000.);
            fitter->second->plotsimple(namesimple_dydz.str(), &rootDirectory, MuonResiduals6DOFFitter::kResSlopeY,1000.);

            fitter->second->plotweighted(nameweighted_x.str(), &rootDirectory, MuonResiduals6DOFFitter::kResidX, MuonResiduals6DOFFitter::kRedChi2, 10.);
            fitter->second->plotweighted(nameweighted_y.str(), &rootDirectory, MuonResiduals6DOFFitter::kResidY, MuonResiduals6DOFFitter::kRedChi2, 10.);
            fitter->second->plotweighted(nameweighted_dxdz.str(), &rootDirectory, MuonResiduals6DOFFitter::kResSlopeX, MuonResiduals6DOFFitter::kRedChi2, 1000.);
            fitter->second->plotweighted(nameweighted_dydz.str(), &rootDirectory, MuonResiduals6DOFFitter::kResSlopeY, MuonResiduals6DOFFitter::kRedChi2, 1000.);
          }

          if (successful_fit)
          {
            if (align_x) params[paramIndex[0]] = deltax_value;
            if (align_y) params[paramIndex[1]] = deltay_value;
            if (align_z) params[paramIndex[2]] = deltaz_value;
            if (align_phix) params[paramIndex[3]] = deltaphix_value;
            if (align_phiy) params[paramIndex[4]] = deltaphiy_value;
            if (align_phiz) params[paramIndex[5]] = deltaphiz_value;
          }
        } // end if 6DOF

        else if (fitter->second->type() == MuonResidualsFitter::k6DOFrphi)
        {
          double deltax_value = fitter->second->value(MuonResiduals6DOFrphiFitter::kAlignX);
          double deltax_error = fitter->second->errorerror(MuonResiduals6DOFrphiFitter::kAlignX);
          double deltax_antisym = fitter->second->antisym(MuonResiduals6DOFrphiFitter::kAlignX);

          double deltay_value = fitter->second->value(MuonResiduals6DOFrphiFitter::kAlignY);
          double deltay_error = fitter->second->errorerror(MuonResiduals6DOFrphiFitter::kAlignY);
          double deltay_antisym = fitter->second->antisym(MuonResiduals6DOFrphiFitter::kAlignY);

          double deltaz_value = fitter->second->value(MuonResiduals6DOFrphiFitter::kAlignZ);
          double deltaz_error = fitter->second->errorerror(MuonResiduals6DOFrphiFitter::kAlignZ);
          double deltaz_antisym = fitter->second->antisym(MuonResiduals6DOFrphiFitter::kAlignZ);

          double deltaphix_value = fitter->second->value(MuonResiduals6DOFrphiFitter::kAlignPhiX);
          double deltaphix_error = fitter->second->errorerror(MuonResiduals6DOFrphiFitter::kAlignPhiX);
          double deltaphix_antisym = fitter->second->antisym(MuonResiduals6DOFrphiFitter::kAlignPhiX);

          double deltaphiy_value = fitter->second->value(MuonResiduals6DOFrphiFitter::kAlignPhiY);
          double deltaphiy_error = fitter->second->errorerror(MuonResiduals6DOFrphiFitter::kAlignPhiY);
          double deltaphiy_antisym = fitter->second->antisym(MuonResiduals6DOFrphiFitter::kAlignPhiY);

          double deltaphiz_value = fitter->second->value(MuonResiduals6DOFrphiFitter::kAlignPhiZ);
          double deltaphiz_error = fitter->second->errorerror(MuonResiduals6DOFrphiFitter::kAlignPhiZ);
          double deltaphiz_antisym = fitter->second->antisym(MuonResiduals6DOFrphiFitter::kAlignPhiZ);

          double sigmaresid_value = fitter->second->value(MuonResiduals6DOFrphiFitter::kResidSigma);
          double sigmaresid_error = fitter->second->errorerror(MuonResiduals6DOFrphiFitter::kResidSigma);
          double sigmaresid_antisym = fitter->second->antisym(MuonResiduals6DOFrphiFitter::kResidSigma);

          double sigmaresslope_value = fitter->second->value(MuonResiduals6DOFrphiFitter::kResSlopeSigma);
          double sigmaresslope_error = fitter->second->errorerror(MuonResiduals6DOFrphiFitter::kResSlopeSigma);
          double sigmaresslope_antisym = fitter->second->antisym(MuonResiduals6DOFrphiFitter::kResSlopeSigma);

          double gammaresid_value, gammaresid_error, gammaresid_antisym, gammaresslope_value, gammaresslope_error, gammaresslope_antisym;
          gammaresid_value = gammaresid_error = gammaresid_antisym = gammaresslope_value = gammaresslope_error = gammaresslope_antisym = 0.;
          if (fitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian &&
              fitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian2D &&
              fitter->second->residualsModel() != MuonResidualsFitter::kGaussPowerTails)
          {
            gammaresid_value = fitter->second->value(MuonResiduals6DOFrphiFitter::kResidGamma);
            gammaresid_error = fitter->second->errorerror(MuonResiduals6DOFrphiFitter::kResidGamma);
            gammaresid_antisym = fitter->second->antisym(MuonResiduals6DOFrphiFitter::kResidGamma);

            gammaresslope_value = fitter->second->value(MuonResiduals6DOFrphiFitter::kResSlopeGamma);
            gammaresslope_error = fitter->second->errorerror(MuonResiduals6DOFrphiFitter::kResSlopeGamma);
            gammaresslope_antisym = fitter->second->antisym(MuonResiduals6DOFrphiFitter::kResSlopeGamma);
          }

          if (writeReport)
          {
            report << "reports[-1].fittype = \"6DOFrphi\"" << std::endl;
            report << "reports[-1].add_parameters(ValErr(" << deltax_value << ", " << deltax_error << ", " << deltax_antisym << "), \\" << std::endl
                << "                           ValErr(" << deltay_value << ", " << deltay_error << ", " << deltay_antisym << "), \\" << std::endl
                << "                           ValErr(" << deltaz_value << ", " << deltaz_error << ", " << deltaz_antisym << "), \\" << std::endl
                << "                           ValErr(" << deltaphix_value << ", " << deltaphix_error << ", " << deltaphix_antisym << "), \\" << std::endl
                << "                           ValErr(" << deltaphiy_value << ", " << deltaphiy_error << ", " << deltaphiy_antisym << "), \\" << std::endl
                << "                           ValErr(" << deltaphiz_value << ", " << deltaphiz_error << ", " << deltaphiz_antisym << "), \\" << std::endl
                << "                           " << loglikelihood << ", " << numsegments << ", " << sumofweights << ", " << redchi2 << ")" << std::endl;
            report << "reports[-1].sigmaresid = ValErr(" << sigmaresid_value << ", " << sigmaresid_error << ", " << sigmaresid_antisym << ")" << std::endl;
            report << "reports[-1].sigmaresslope = ValErr(" << sigmaresslope_value << ", " << sigmaresslope_error << ", " << sigmaresslope_antisym << ")" << std::endl;
            if (fitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian &&
                fitter->second->residualsModel() != MuonResidualsFitter::kPureGaussian2D &&
                fitter->second->residualsModel() != MuonResidualsFitter::kGaussPowerTails)
            {
              report << "reports[-1].gammaresid = ValErr(" << gammaresid_value << ", " << gammaresid_error << ", " << gammaresid_antisym << ")" << std::endl;
              report << "reports[-1].gammaresslope = ValErr(" << gammaresslope_value << ", " << gammaresslope_error << ", " << gammaresslope_antisym << ")" << std::endl;
            }

            report << "reports[-1].add_stats(" << fitter->second->median(MuonResiduals6DOFrphiFitter::kResid) << ", " << "None, "
                << fitter->second->median(MuonResiduals6DOFrphiFitter::kResSlope) << ", " << "None, "
                << fitter->second->mean(MuonResiduals6DOFrphiFitter::kResid, 30.) << ", " << "None, "
                << fitter->second->mean(MuonResiduals6DOFrphiFitter::kResSlope, 20.) << ", " << "None, "
                << fitter->second->mean(MuonResiduals6DOFrphiFitter::kResid, 15.) << ", " << "None, "
                << fitter->second->mean(MuonResiduals6DOFrphiFitter::kResSlope, 10.) << ", " << "None, "
                << fitter->second->wmean(MuonResiduals6DOFrphiFitter::kResid, MuonResiduals6DOFrphiFitter::kRedChi2, 30.) << ", " << "None, "
                << fitter->second->wmean(MuonResiduals6DOFrphiFitter::kResSlope, MuonResiduals6DOFrphiFitter::kRedChi2, 20.) << ", " << "None, "
                << fitter->second->wmean(MuonResiduals6DOFrphiFitter::kResid, MuonResiduals6DOFrphiFitter::kRedChi2, 15.) << ", " << "None, "
                << fitter->second->wmean(MuonResiduals6DOFrphiFitter::kResSlope, MuonResiduals6DOFrphiFitter::kRedChi2, 10.) << ", " << "None, "
                << fitter->second->stdev(MuonResiduals6DOFrphiFitter::kResid, 30.) << ", " << "None, "
                << fitter->second->stdev(MuonResiduals6DOFrphiFitter::kResSlope, 20.) << ", " << "None, "
                << fitter->second->stdev(MuonResiduals6DOFrphiFitter::kResid, 15.) << ", " << "None, "
                << fitter->second->stdev(MuonResiduals6DOFrphiFitter::kResSlope, 10.) << ", " << "None)" << std::endl;

            std::stringstream namesimple_x, namesimple_dxdz, nameweighted_x, nameweighted_dxdz;
            namesimple_x << cname << "_simple_x";
            namesimple_dxdz << cname << "_simple_dxdz";
            nameweighted_x << cname << "_weighted_x";
            nameweighted_dxdz << cname << "_weighted_dxdz";

            fitter->second->plotsimple(namesimple_x.str(), &rootDirectory, MuonResiduals6DOFrphiFitter::kResid, 10.);
            fitter->second->plotsimple(namesimple_dxdz.str(), &rootDirectory, MuonResiduals6DOFrphiFitter::kResSlope, 1000.);

            fitter->second->plotweighted(nameweighted_x.str(), &rootDirectory, MuonResiduals6DOFrphiFitter::kResid, MuonResiduals6DOFrphiFitter::kRedChi2, 10.);
            fitter->second->plotweighted(nameweighted_dxdz.str(), &rootDirectory, MuonResiduals6DOFrphiFitter::kResSlope, MuonResiduals6DOFrphiFitter::kRedChi2, 1000.);
          }

          if (successful_fit)
          {
            if (align_x) params[paramIndex[0]] = deltax_value;
            if (align_y) params[paramIndex[1]] = deltay_value;
            if (align_z) params[paramIndex[2]] = deltaz_value;
            if (align_phix) params[paramIndex[3]] = deltaphix_value;
            if (align_phiy) params[paramIndex[4]] = deltaphiy_value;
            if (align_phiz) params[paramIndex[5]] = deltaphiz_value;
          }
        } // end if 6DOFrphi

        if (successful_fit)
        {
          std::vector<Alignable*> oneortwo;
          oneortwo.push_back(*ali);
          if (thisali != *ali) oneortwo.push_back(thisali);
          m_alignmentParameterStore->setAlignmentPositionError(oneortwo, 0., 0.);
        }
        else
        {
          std::cout << "MINUIT fit failed!" << std::endl;
          if (writeReport)
          {
            report << "reports[-1].status = \"MINUITFAIL\"" << std::endl;
          }

          for (int i = 0; i < numParams; i++)  cov[i][i] = 1000.;

          std::vector<Alignable*> oneortwo;
          oneortwo.push_back(*ali);
          if (thisali != *ali) oneortwo.push_back(thisali);
          m_alignmentParameterStore->setAlignmentPositionError(oneortwo, 1000., 0.);
        }
      }
      else
      { // too few hits
        std::cout << "Too few hits!" << std::endl;
        if (writeReport)
        {
          report << "reports[-1].status = \"TOOFEWHITS\"" << std::endl;
        }

        for (int i = 0; i < numParams; i++)  cov[i][i] = 1000.;

        std::vector<Alignable*> oneortwo;
        oneortwo.push_back(*ali);
        if (thisali != *ali) oneortwo.push_back(thisali);
        m_alignmentParameterStore->setAlignmentPositionError(oneortwo, 1000., 0.);
      }

      AlignmentParameters *parnew = (*ali)->alignmentParameters()->cloneFromSelected(params, cov);
      (*ali)->setAlignmentParameters(parnew);
      m_alignmentParameterStore->applyParameters(*ali);
      (*ali)->alignmentParameters()->setValid(true);

      std::cout << cname<<" fittime= "<< stop_watch.CpuTime() << " sec" << std::endl;
    } // end we have a fitter for this alignable

    if (writeReport) report << std::endl;

  } // end loop over alignables

  if (writeReport) report.close();
}


void MuonAlignmentFromReference::readTmpFiles()
{
  for (std::vector<std::string>::const_iterator fileName = m_readTemporaryFiles.begin();  fileName != m_readTemporaryFiles.end();  ++fileName)
  {
    FILE *file;
    int size;
    file = fopen(fileName->c_str(), "r");
    if (file == NULL)
      throw cms::Exception("MuonAlignmentFromReference") << "file \"" << *fileName << " can't be opened (doesn't exist?)" << std::endl;

    fread(&size, sizeof(int), 1, file);
    if (int(m_indexes.size()) != size)
      throw cms::Exception("MuonAlignmentFromReference") << "file \"" << *fileName << "\" has " << size
        << " fitters, but this job has " << m_indexes.size() << " fitters (probably corresponds to the wrong alignment job)" << std::endl;

    int i = 0;
    for (std::vector<unsigned int>::const_iterator index = m_indexes.begin();  index != m_indexes.end();  ++index, ++i)
    {
      MuonResidualsTwoBin *fitter = m_fitterOrder[*index];
      unsigned int index_toread;
      fread(&index_toread, sizeof(unsigned int), 1, file);
      if (*index != index_toread)
        throw cms::Exception("MuonAlignmentFromReference") << "file \"" << *fileName << "\" has index " << index_toread
          << " at position " << i << ", but this job is expecting " << *index << " (probably corresponds to the wrong alignment job)" << std::endl;
      fitter->read(file, i);
    }

    fclose(file);
  }
}


void MuonAlignmentFromReference::writeTmpFiles()
{
  FILE *file;
  file = fopen(m_writeTemporaryFile.c_str(), "w");
  int size = m_indexes.size();
  fwrite(&size, sizeof(int), 1, file);

  int i = 0;
  for (std::vector<unsigned int>::const_iterator index = m_indexes.begin();  index != m_indexes.end();  ++index, ++i)
  {
    MuonResidualsTwoBin *fitter = m_fitterOrder[*index];
    unsigned int index_towrite = *index;
    fwrite(&index_towrite, sizeof(unsigned int), 1, file);
    fitter->write(file, i);
  }

  fclose(file);
}

void MuonAlignmentFromReference::correctBField()
{
  for (std::vector<unsigned int>::const_iterator index = m_indexes.begin();  index != m_indexes.end();  ++index)
  {
    std::cout<<"correcting B in "<<chamberPrettyNameFromId(*index)<<std::endl;
    MuonResidualsTwoBin *fitter = m_fitterOrder[*index];
    fitter->correctBField();
  }
}


void MuonAlignmentFromReference::eraseNotSelectedResiduals()
{
  for (std::vector<unsigned int>::const_iterator index = m_indexes.begin();  index != m_indexes.end();  ++index)
  {
    std::cout<<"erasing in "<<chamberPrettyNameFromId(*index)<<std::endl;
    MuonResidualsTwoBin *fitter = m_fitterOrder[*index];
    fitter->eraseNotSelectedResiduals();
  }
}


void MuonAlignmentFromReference::selectResidualsPeaks()
{
  // should not be called with negative peakNSigma
  assert(m_peakNSigma>0.);
  
  for (std::vector<unsigned int>::const_iterator index = m_indexes.begin();  index != m_indexes.end();  ++index)
  {
    MuonResidualsTwoBin *fitter = m_fitterOrder[*index];

    int nvar = 2;
    int vars_index[10] = {0,1};
    if (fitter->type() == MuonResidualsFitter::k5DOF)
    {
      if (fitter->useRes() == MuonResidualsFitter::k1111 || fitter->useRes() == MuonResidualsFitter::k1110 || fitter->useRes() == MuonResidualsFitter::k1010) {
        nvar = 2;
        vars_index[0] = MuonResiduals5DOFFitter::kResid;
        vars_index[1] = MuonResiduals5DOFFitter::kResSlope;
      }
      else if (fitter->useRes() == MuonResidualsFitter::k1100) {
        nvar = 1;
        vars_index[0] = MuonResiduals5DOFFitter::kResid;
      }
      else if (fitter->useRes() == MuonResidualsFitter::k0010) {
        nvar = 1;
        vars_index[0] = MuonResiduals5DOFFitter::kResSlope;
      }
    }
    else if (fitter->type() == MuonResidualsFitter::k6DOF)
    {
      if (fitter->useRes() == MuonResidualsFitter::k1111) {
        nvar = 4;
        vars_index[0] = MuonResiduals6DOFFitter::kResidX;
        vars_index[1] = MuonResiduals6DOFFitter::kResidY;
        vars_index[2] = MuonResiduals6DOFFitter::kResSlopeX;
        vars_index[3] = MuonResiduals6DOFFitter::kResSlopeY;
      }
      else if (fitter->useRes() == MuonResidualsFitter::k1110) {
        nvar = 3;
        vars_index[0] = MuonResiduals6DOFFitter::kResidX;
        vars_index[1] = MuonResiduals6DOFFitter::kResidY;
        vars_index[2] = MuonResiduals6DOFFitter::kResSlopeX;
      }
      else if (fitter->useRes() == MuonResidualsFitter::k1010) {
        nvar = 2;
        vars_index[0] = MuonResiduals6DOFFitter::kResidX;
        vars_index[2] = MuonResiduals6DOFFitter::kResSlopeX;
      }
      else if (fitter->useRes() == MuonResidualsFitter::k1100) {
        nvar = 2;
        vars_index[0] = MuonResiduals6DOFFitter::kResidX;
        vars_index[1] = MuonResiduals6DOFFitter::kResidY;
      }
      else if (fitter->useRes() == MuonResidualsFitter::k0010) {
        nvar = 1;
        vars_index[0] = MuonResiduals6DOFFitter::kResSlopeX;
      }
    }
    else if (fitter->type() == MuonResidualsFitter::k6DOFrphi)
    {
      if (fitter->useRes() == MuonResidualsFitter::k1111 || fitter->useRes() == MuonResidualsFitter::k1110 || fitter->useRes() == MuonResidualsFitter::k1010) {
        nvar = 2;
        vars_index[0] = MuonResiduals6DOFrphiFitter::kResid;
        vars_index[1] = MuonResiduals6DOFrphiFitter::kResSlope;
      }
      else if (fitter->useRes() == MuonResidualsFitter::k1100) {
        nvar = 1;
        vars_index[0] = MuonResiduals6DOFrphiFitter::kResid;
      }
      else if (fitter->useRes() == MuonResidualsFitter::k0010) {
        nvar = 1;
        vars_index[0] = MuonResiduals6DOFrphiFitter::kResSlope;
      }
    }
    else assert(false);

    std::cout<<"selecting in "<<chamberPrettyNameFromId(*index)<<std::endl;
    
    fitter->selectPeakResiduals(m_peakNSigma, nvar, vars_index);
  }
}


std::string MuonAlignmentFromReference::chamberPrettyNameFromId(unsigned int idx)
{
   DetId id(idx); 
   char cname[40];
   if (id.subdetId() == MuonSubdetId::DT)
   {
     DTChamberId chamberId(id.rawId());
     sprintf(cname, "MB%+d/%d/%02d", chamberId.wheel(), chamberId.station(), chamberId.sector());
   }
   else if (id.subdetId() == MuonSubdetId::CSC)
   {
     CSCDetId chamberId(id.rawId());
     sprintf(cname, "ME%s%d/%d/%02d", (chamberId.endcap() == 1 ? "+" : "-"), chamberId.station(), chamberId.ring(), chamberId.chamber());
   }
   return std::string(cname);
}


void MuonAlignmentFromReference::fillNtuple()
{
  // WARNING: does not support two bin option!!!

  for (std::vector<unsigned int>::const_iterator index = m_indexes.begin();  index != m_indexes.end();  ++index)
  {
    DetId detid(*index);
    if (detid.det() != DetId::Muon || !( detid.subdetId() == MuonSubdetId::DT || detid.subdetId() == MuonSubdetId::CSC) ) assert(false);

    if(detid.subdetId() == MuonSubdetId::DT)
    {
      m_tree_row.is_dt = (Bool_t) true;
      DTChamberId id(*index);
      m_tree_row.is_plus = (Bool_t) true;
      m_tree_row.station = (UChar_t) id.station();
      m_tree_row.ring_wheel = (Char_t) id.wheel();
      m_tree_row.sector = (UChar_t) id.sector();
    }
    else
    {
      m_tree_row.is_dt = (Bool_t) false;
      CSCDetId id(*index);
      m_tree_row.is_plus = (Bool_t) (id.endcap() == 1);
      m_tree_row.station = (UChar_t) id.station();
      m_tree_row.ring_wheel = (Char_t) id.ring();
      m_tree_row.sector = (UChar_t) id.chamber();
    }

    MuonResidualsTwoBin *fitter = m_fitterOrder[*index];

    std::vector<double*>::const_iterator residual = fitter->residualsPos_begin();
    std::vector<bool>::const_iterator residual_ok = fitter->residualsPos_ok_begin();
    for (;  residual != fitter->residualsPos_end();  ++residual, ++residual_ok)
    {
      if (fitter->type() == MuonResidualsFitter::k5DOF || fitter->type() == MuonResidualsFitter::k6DOFrphi)
      {
        m_tree_row.res_x       = (Float_t) (*residual)[MuonResiduals5DOFFitter::kResid];
        m_tree_row.res_y       = (Float_t) 0.;
        m_tree_row.res_slope_x = (Float_t) (*residual)[MuonResiduals5DOFFitter::kResSlope];
        m_tree_row.res_slope_y = (Float_t) 0.;
        m_tree_row.pos_x       = (Float_t) (*residual)[MuonResiduals5DOFFitter::kPositionX];
        m_tree_row.pos_y       = (Float_t) (*residual)[MuonResiduals5DOFFitter::kPositionY];
        m_tree_row.angle_x     = (Float_t) (*residual)[MuonResiduals5DOFFitter::kAngleX];
        m_tree_row.angle_y     = (Float_t) (*residual)[MuonResiduals5DOFFitter::kAngleY];
        m_tree_row.pz          = (Float_t) (*residual)[MuonResiduals5DOFFitter::kPz];
        m_tree_row.pt          = (Float_t) (*residual)[MuonResiduals5DOFFitter::kPt];
        m_tree_row.q           = (Char_t) (*residual)[MuonResiduals5DOFFitter::kCharge];
        m_tree_row.select      = (Bool_t) *residual_ok;
      }
      else if (fitter->type() == MuonResidualsFitter::k6DOF)
      {
        m_tree_row.res_x       = (Float_t) (*residual)[MuonResiduals6DOFFitter::kResidX];
        m_tree_row.res_y       = (Float_t) (*residual)[MuonResiduals6DOFFitter::kResidY];
        m_tree_row.res_slope_x = (Float_t) (*residual)[MuonResiduals6DOFFitter::kResSlopeX];
        m_tree_row.res_slope_y = (Float_t) (*residual)[MuonResiduals6DOFFitter::kResSlopeY];
        m_tree_row.pos_x       = (Float_t) (*residual)[MuonResiduals6DOFFitter::kPositionX];
        m_tree_row.pos_y       = (Float_t) (*residual)[MuonResiduals6DOFFitter::kPositionY];
        m_tree_row.angle_x     = (Float_t) (*residual)[MuonResiduals6DOFFitter::kAngleX];
        m_tree_row.angle_y     = (Float_t) (*residual)[MuonResiduals6DOFFitter::kAngleY];
        m_tree_row.pz          = (Float_t) (*residual)[MuonResiduals6DOFFitter::kPz];
        m_tree_row.pt          = (Float_t) (*residual)[MuonResiduals6DOFFitter::kPt];
        m_tree_row.q           = (Char_t) (*residual)[MuonResiduals6DOFFitter::kCharge];
        m_tree_row.select      = (Bool_t) *residual_ok;
      }
      else assert(false);

      m_ttree->Fill();
    }
  }
}


#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
DEFINE_EDM_PLUGIN(AlignmentAlgorithmPluginFactory, MuonAlignmentFromReference, "MuonAlignmentFromReference");
