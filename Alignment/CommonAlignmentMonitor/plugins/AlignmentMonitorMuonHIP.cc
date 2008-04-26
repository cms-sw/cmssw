// -*- C++ -*-
//
// Package:     CommonAlignmentProducer
// Class  :     AlignmentMonitorMuonHIP
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Thu Jun 28 01:38:33 CDT 2007
//

// system include files
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h" 
#include <DataFormats/GeometrySurface/interface/LocalError.h> 
#include "TH1.h" 
#include "TObject.h" 
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/MuonAlignment/interface/AlignableDTWheel.h"
#include "Alignment/MuonAlignment/interface/AlignableDTChamber.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCStation.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCChamber.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "Alignment/HIPAlignmentAlgorithm/interface/HIPUserVariables.h"

#include "TProfile.h"
#include "TTree.h"

#include <fstream>

// user include files

// 
// class definition
// 

class AlignmentMonitorMuonHIP: public AlignmentMonitorBase {
   public:
      AlignmentMonitorMuonHIP(const edm::ParameterSet& cfg);
      ~AlignmentMonitorMuonHIP() {};

      void book();
      void event(const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& iTrajTracks);
      void afterAlignment(const edm::EventSetup &iSetup);

   private:
      typedef TH1F* TH1FPtr;
      typedef TProfile* TProfilePtr;

      // Histogram parameters
      edm::ParameterSet m_params;
      int m_params_iterations;
      double m_params_iterations_low, m_params_iterations_high;
      int m_params_bins;
      double m_params_xresid_low, m_params_xresid_high, m_params_xresidwide_low, m_params_xresidwide_high, m_params_yresid_low, m_params_yresid_high;
      double m_params_xDT_low, m_params_xDT_high, m_params_yDT_low, m_params_yDT_high, m_params_xCSC_low, m_params_xCSC_high, m_params_yCSC_low, m_params_yCSC_high;
      double m_params_xpull_low, m_params_xpull_high, m_params_ypull_low, m_params_ypull_high;

      edm::ParameterSet m_book;
      std::string m_book_mode;
      bool m_book_nhits_vsiter, m_book_conv_x, m_book_conv_y, m_book_conv_z, m_book_conv_phix, m_book_conv_phiy, m_book_conv_phiz;
      bool m_book_xresid, m_book_xresidwide, m_book_yresid;
      bool m_book_wxresid, m_book_wxresidwide, m_book_wyresid;
      bool m_book_wxresid_vsx, m_book_wxresid_vsy, m_book_wyresid_vsx, m_book_wyresid_vsy;
      bool m_book_xpull, m_book_ypull;
      bool m_book_before, m_book_after;
      
      bool m_createPythonGeometry;

      // Indexing for fast lookup in event loop
      std::map<Alignable*, unsigned int> m_disk_index, m_chamber_index, m_layer_index;
      unsigned int m_numHistograms;
      std::map<Alignable*, int> m_rawid;

      // Histograms that are the same for each iteration
      TH1FPtr *m_nhits_vsiter, *m_conv_x, *m_conv_y, *m_conv_z, *m_conv_phix, *m_conv_phiy, *m_conv_phiz;

      // Histograms that are different for each iteration
      TH1FPtr *m_xresid, *m_xresidwide, *m_yresid;
      TH1FPtr *m_wxresid, *m_wxresidwide, *m_wyresid;
      TProfilePtr *m_wxresid_vsx, *m_wxresid_vsy, *m_wyresid_vsx, *m_wyresid_vsy;
      TH1FPtr *m_xpull, *m_ypull;

      // The ntuple (ntuples MUST be different for each iteration, or you'll get problems declaring branches)
      TTree *m_before, *m_after;
      Int_t m_before_rawid, m_before_level;
      Float_t m_before_x, m_before_y, m_before_z, m_before_phix, m_before_phiy, m_before_phiz;
      Int_t m_after_rawid, m_after_level;
      Float_t m_after_x, m_after_y, m_after_z, m_after_phix, m_after_phiy, m_after_phiz;
      Float_t m_after_xerr, m_after_yerr, m_after_zerr, m_after_phixerr, m_after_phiyerr, m_after_phizerr;      

      // Private functions
      void createPythonGeometry();
      void bookByAli(const char *level, const int rawid, const unsigned int index);
      void fill(unsigned int index, double x_residual, double y_residual, double x_reserr2, double y_reserr2, double xpos, double ypos, bool y_valid);
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

// AlignmentMonitorMuonHIP::AlignmentMonitorMuonHIP(const AlignmentMonitorMuonHIP& rhs)
// {
//    // do actual copying here;
// }

//
// assignment operators
//
// const AlignmentMonitorMuonHIP& AlignmentMonitorMuonHIP::operator=(const AlignmentMonitorMuonHIP& rhs)
// {
//   //An exception safe implementation is
//   AlignmentMonitorMuonHIP temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

AlignmentMonitorMuonHIP::AlignmentMonitorMuonHIP(const edm::ParameterSet& cfg)
   : AlignmentMonitorBase(cfg, "AlignmentMonitorMuonHIP")
   , m_params(cfg.getParameter<edm::ParameterSet>("params"))
   , m_book(cfg.getParameter<edm::ParameterSet>("book"))
   , m_createPythonGeometry(cfg.getParameter<bool>("createPythonGeometry"))
{
   m_params_iterations = m_params.getParameter<unsigned int>("iterations");
   m_params_iterations_low = m_params.getParameter<double>("iterations_low");
   m_params_iterations_high = m_params.getParameter<double>("iterations_high");
   m_params_bins = m_params.getParameter<unsigned int>("bins");
   m_params_xresid_low = m_params.getParameter<double>("xresid_low");
   m_params_xresid_high = m_params.getParameter<double>("xresid_high");
   m_params_xresidwide_low = m_params.getParameter<double>("xresidwide_low");
   m_params_xresidwide_high = m_params.getParameter<double>("xresidwide_high");
   m_params_yresid_low = m_params.getParameter<double>("yresid_low");
   m_params_yresid_high = m_params.getParameter<double>("yresid_high");
   m_params_xDT_low = m_params.getParameter<double>("xDT_low");
   m_params_xDT_high = m_params.getParameter<double>("xDT_high");
   m_params_yDT_low = m_params.getParameter<double>("yDT_low");
   m_params_yDT_high = m_params.getParameter<double>("yDT_high");
   m_params_xCSC_low = m_params.getParameter<double>("xCSC_low");
   m_params_xCSC_high = m_params.getParameter<double>("xCSC_high");
   m_params_yCSC_low = m_params.getParameter<double>("yCSC_low");
   m_params_yCSC_high = m_params.getParameter<double>("yCSC_high");
   m_params_xpull_low = m_params.getParameter<double>("xpull_low");
   m_params_xpull_high = m_params.getParameter<double>("xpull_high");
   m_params_ypull_low = m_params.getParameter<double>("ypull_low");
   m_params_ypull_high = m_params.getParameter<double>("ypull_high");

   m_book_mode = m_book.getParameter<std::string>("mode");
   if (m_book_mode != std::string("selected")  &&
       m_book_mode != std::string("disk")  &&
       m_book_mode != std::string("chamber")  &&
       m_book_mode != std::string("layer")) {
      throw cms::Exception("BadConfig") << "AlignmentMonitorMuonHIP.book.mode must be \"selected\", \"disk\", \"chamber\", or \"layer\".";
   }

   m_book_nhits_vsiter = m_book.getParameter<bool>("nhits_vsiter");
   m_book_conv_x = m_book.getParameter<bool>("conv_x");
   m_book_conv_y = m_book.getParameter<bool>("conv_y");
   m_book_conv_z = m_book.getParameter<bool>("conv_z");
   m_book_conv_phix = m_book.getParameter<bool>("conv_phix");
   m_book_conv_phiy = m_book.getParameter<bool>("conv_phiy");
   m_book_conv_phiz = m_book.getParameter<bool>("conv_phiz");
   m_book_xresid = m_book.getParameter<bool>("xresid");
   m_book_xresidwide = m_book.getParameter<bool>("xresidwide");
   m_book_yresid = m_book.getParameter<bool>("yresid");
   m_book_wxresid = m_book.getParameter<bool>("wxresid");
   m_book_wxresidwide = m_book.getParameter<bool>("wxresidwide");
   m_book_wyresid = m_book.getParameter<bool>("wyresid");
   m_book_wxresid_vsx = m_book.getParameter<bool>("wxresid_vsx");
   m_book_wxresid_vsy = m_book.getParameter<bool>("wxresid_vsy");
   m_book_wyresid_vsx = m_book.getParameter<bool>("wyresid_vsx");
   m_book_wyresid_vsy = m_book.getParameter<bool>("wyresid_vsy");
   m_book_xpull = m_book.getParameter<bool>("xpull");
   m_book_ypull = m_book.getParameter<bool>("ypull");
   m_book_before = m_book.getParameter<bool>("before");
   m_book_after = m_book.getParameter<bool>("after");
}

//
// member functions
//

//////////////////////////////////////////////////////////////////////
// book()
//////////////////////////////////////////////////////////////////////

void AlignmentMonitorMuonHIP::book() {
   //////////////////////////////////////////////////////////////////////
   // Make histograms based on the selected alignables.  This will make
   // files for layer-by-layer alignments huge and disk-by-wheel
   // alignments tiny.
   //////////////////////////////////////////////////////////////////////

   unsigned int index = 0;  // merge EVERYTHING (good for a quick look in the TBrowser)
   index++;

   std::vector<Alignable*> alignables;
   if (m_book_mode == std::string("selected")) {
      alignables = pStore()->alignables();
   }
   else if (m_book_mode == std::string("disk")) {
      alignables = pMuon()->DTWheels();

      std::vector<Alignable*> more = pMuon()->CSCStations();
      for (std::vector<Alignable*>::const_iterator miter = more.begin();  miter != more.end();  ++miter) {
	 alignables.push_back(*miter);
      }
   }
   else if (m_book_mode == std::string("chamber")) {
      alignables = pMuon()->DTChambers();

      std::vector<Alignable*> more = pMuon()->CSCChambers();
      for (std::vector<Alignable*>::const_iterator miter = more.begin();  miter != more.end();  ++miter) {
	 alignables.push_back(*miter);
      }
   }
   else if (m_book_mode == std::string("layer")) {
      alignables = pMuon()->DTSuperLayers();

      std::vector<Alignable*> more = pMuon()->CSCLayers();
      for (std::vector<Alignable*>::const_iterator miter = more.begin();  miter != more.end();  ++miter) {
	 alignables.push_back(*miter);
      }
   }

   for (std::vector<Alignable*>::const_iterator aliiter = alignables.begin();  aliiter != alignables.end();  ++aliiter) {
      Alignable *ali = *aliiter;

      // Sorry, histogramming code needs to be very specific.  If the
      // plots were abstract and generic, what could we learn from them?
      if (ali->mother() != NULL) {
	 AlignableDTChamber *dtSuperLayer_mother = dynamic_cast<AlignableDTChamber*>(ali->mother());
	 AlignableCSCChamber *cscLayer_mother = dynamic_cast<AlignableCSCChamber*>(ali->mother());
	 if (dtSuperLayer_mother  ||  cscLayer_mother) {
	    m_layer_index[ali] = index;
	    m_rawid[ali] = ali->geomDetId().rawId();
	    index++;
	    ali = ali->mother();
	 }
      }

      AlignableDTChamber *dtChamber = dynamic_cast<AlignableDTChamber*>(ali);
      AlignableCSCChamber *cscChamber = dynamic_cast<AlignableCSCChamber*>(ali);
      if (dtChamber  ||  cscChamber) {
	 m_chamber_index[ali] = index;
	 m_rawid[ali] = ali->geomDetId().rawId();
	 index++;
	 ali = ali->mother();

	 if (dtChamber) ali = ali->mother();  // Skip station level
	 // if (cscChamber) ali = ali->mother();  // No (real) station level to skip (yet?)
      }

      AlignableDTWheel *dtWheel = dynamic_cast<AlignableDTWheel*>(ali);
      AlignableCSCStation *cscDisk = dynamic_cast<AlignableCSCStation*>(ali);
	 
      if (dtWheel  ||  cscDisk) {
	 m_disk_index[ali] = index;

	 Alignable *descend = ali;
	 while (true) {
	    std::vector<Alignable*> components = descend->components();
	    if (components.size() > 0) {
	       descend = components[0];
	       m_rawid[ali] = descend->geomDetId().rawId();
	       if (m_rawid[ali] != 0) break;
	    }
	    else {
	       edm::LogError("AlignmentMonitorMuonHIP")
		  << "Something happened to the topology of the Alignable tree: I find a" << std::endl
		  << "DTWheel or a CSCDisk (a.k.a. CSCStation) without components containing DetIds" << std::endl;
	       static int wheeldiskNumber = 0;
	       wheeldiskNumber++;
	       m_rawid[ali] = wheeldiskNumber;
	    }
	 }

	 index++;
	 ali = ali->mother();
      }
   }
   m_numHistograms = index;

   if (m_numHistograms == 0) {
      edm::LogError("AlignmentMonitorMuonHIP")
	 << "================================================================================" << std::endl
	 << "====     Generating no (zero) histograms (nada, the big zip-a-dee-doo-da)    ===" << std::endl
	 << "====                       (is this what you want?!?!?)                      ===" << std::endl
	 << "================================================================================" << std::endl;
   }

   //////////////////////////////////////////////////////////////////////
   // Now we know how many histograms there are and have a quick-lookup
   //////////////////////////////////////////////////////////////////////

   m_nhits_vsiter = new TH1FPtr[m_numHistograms];
   m_conv_x = new TH1FPtr[m_numHistograms];
   m_conv_y = new TH1FPtr[m_numHistograms];
   m_conv_z = new TH1FPtr[m_numHistograms];
   m_conv_phix = new TH1FPtr[m_numHistograms];
   m_conv_phiy = new TH1FPtr[m_numHistograms];
   m_conv_phiz = new TH1FPtr[m_numHistograms];
   m_xresid = new TH1FPtr[m_numHistograms];
   m_xresidwide = new TH1FPtr[m_numHistograms];
   m_yresid = new TH1FPtr[m_numHistograms];
   m_wxresid = new TH1FPtr[m_numHistograms];
   m_wxresidwide = new TH1FPtr[m_numHistograms];
   m_wyresid = new TH1FPtr[m_numHistograms];
   m_wxresid_vsx = new TProfilePtr[m_numHistograms];
   m_wxresid_vsy = new TProfilePtr[m_numHistograms];
   m_wyresid_vsx = new TProfilePtr[m_numHistograms];
   m_wyresid_vsy = new TProfilePtr[m_numHistograms];
   m_xpull = new TH1FPtr[m_numHistograms];
   m_ypull = new TH1FPtr[m_numHistograms];

   //////////////////////////////////////////////////////////////////////
   // These three loops book most of the histograms
   //////////////////////////////////////////////////////////////////////

   for (std::map<Alignable*, unsigned int>::const_iterator aliint = m_disk_index.begin();  aliint != m_disk_index.end();  ++aliint) {
      bookByAli("disk", m_rawid[aliint->first], aliint->second);
   }      

   for (std::map<Alignable*, unsigned int>::const_iterator aliint = m_chamber_index.begin();  aliint != m_chamber_index.end();  ++aliint) {
      bookByAli("chamber", m_rawid[aliint->first], aliint->second);
   }      

   for (std::map<Alignable*, unsigned int>::const_iterator aliint = m_layer_index.begin();  aliint != m_layer_index.end();  ++aliint) {
      bookByAli("layer", m_rawid[aliint->first], aliint->second);
   }      

   //////////////////////////////////////////////////////////////////////
   // Handle the zero case (plot everything) for *some* histograms
   //////////////////////////////////////////////////////////////////////

   m_nhits_vsiter[0] = NULL;
   m_conv_x[0] = NULL;
   m_conv_y[0] = NULL;
   m_conv_z[0] = NULL;
   m_conv_phix[0] = NULL;
   m_conv_phiy[0] = NULL;
   m_conv_phiz[0] = NULL;

   char dir[256], name[256], title[256];
   sprintf(dir, "/iterN/");
   sprintf(name, "xresid");
   sprintf(title, "x residual for iteration %d", iteration());
   m_xresid[0] = book1D(dir, name, title, m_params_bins, m_params_xresid_low, m_params_xresid_high);

   sprintf(dir, "/iterN/");
   sprintf(name, "xresidwide");
   sprintf(title, "x residual for iteration %d", iteration());
   m_xresidwide[0] = book1D(dir, name, title, m_params_bins, m_params_xresidwide_low, m_params_xresidwide_high);

   sprintf(dir, "/iterN/");
   sprintf(name, "yresid");
   sprintf(title, "y residual for iteration %d", iteration());
   m_yresid[0] = book1D(dir, name, title, m_params_bins, m_params_yresid_low, m_params_yresid_high);

   sprintf(dir, "/iterN/");
   sprintf(name, "wxresid");
   sprintf(title, "Weighted x residual for iteration %d", iteration());
   m_wxresid[0] = book1D(dir, name, title, m_params_bins, m_params_xresid_low, m_params_xresid_high);

   sprintf(dir, "/iterN/");
   sprintf(name, "wxresidwide");
   sprintf(title, "Weighted x residual for iteration %d", iteration());
   m_wxresidwide[0] = book1D(dir, name, title, m_params_bins, m_params_xresidwide_low, m_params_xresidwide_high);

   sprintf(dir, "/iterN/");
   sprintf(name, "wyresid");
   sprintf(title, "Weighted y residual for iteration %d", iteration());
   m_wyresid[0] = book1D(dir, name, title, m_params_bins, m_params_yresid_low, m_params_yresid_high);

   m_wxresid_vsx[0] = NULL;
   m_wxresid_vsy[0] = NULL;
   m_wyresid_vsx[0] = NULL;
   m_wyresid_vsy[0] = NULL;

   sprintf(dir, "/iterN/");
   sprintf(name, "xpull");
   sprintf(title, "x pull distribution for iteration %d", iteration());
   m_xpull[0] = book1D(dir, name, title, m_params_bins, m_params_xpull_low, m_params_xpull_high);

   sprintf(dir, "/iterN/");
   sprintf(name, "ypull");
   sprintf(title, "y pull distribution for iteration %d", iteration());
   m_ypull[0] = book1D(dir, name, title, m_params_bins, m_params_ypull_low, m_params_ypull_high);

   //////////////////////////////////////////////////////////////////////
   // Finally, book the alignable-wise ntuple and fill "before"
   //////////////////////////////////////////////////////////////////////

   if (m_book_before) {
      m_before = bookTree("/iterN/", "before", "positions before iteration");
      m_before->Branch("rawid", &m_before_rawid, "rawid/I");
      m_before->Branch("level", &m_before_level, "level/I");
      m_before->Branch("x", &m_before_x, "x/F");
      m_before->Branch("y", &m_before_y, "y/F");
      m_before->Branch("z", &m_before_z, "z/F");
      m_before->Branch("phix", &m_before_phix, "phix/F");
      m_before->Branch("phiy", &m_before_phiy, "phiy/F");
      m_before->Branch("phiz", &m_before_phiz, "phiz/F");
   }
   else {
      m_before = NULL;
   }

   if (m_book_after) {
      m_after = bookTree("/iterN/", "after", "positions after iteration");
      m_after->Branch("rawid", &m_after_rawid, "rawid/I");
      m_after->Branch("level", &m_after_level, "level/I");
      m_after->Branch("x", &m_after_x, "x/F");
      m_after->Branch("xerr", &m_after_xerr, "xerr/F");
      m_after->Branch("y", &m_after_y, "y/F");
      m_after->Branch("yerr", &m_after_yerr, "yerr/F");
      m_after->Branch("z", &m_after_z, "z/F");
      m_after->Branch("zerr", &m_after_zerr, "zerr/F");
      m_after->Branch("phix", &m_after_phix, "phix/F");
      m_after->Branch("phixerr", &m_after_phixerr, "phixerr/F");
      m_after->Branch("phiy", &m_after_phiy, "phiy/F");
      m_after->Branch("phiyerr", &m_after_phiyerr, "phiyerr/F");
      m_after->Branch("phiz", &m_after_phiz, "phiz/F");
      m_after->Branch("phizerr", &m_after_phizerr, "phizerr/F");
   }
   else {
      m_after = NULL;
   }

   for (std::vector<Alignable*>::const_iterator aliiter = alignables.begin();  aliiter != alignables.end();  ++aliiter) {
      for (Alignable *ali = *aliiter;  ali != NULL;  ali = ali->mother()) {
	 std::map<Alignable*, unsigned int>::const_iterator disk = m_disk_index.find(ali);
	 std::map<Alignable*, unsigned int>::const_iterator chamber = m_chamber_index.find(ali);
	 std::map<Alignable*, unsigned int>::const_iterator layer = m_layer_index.find(ali);

	 LocalVector displacement = ali->surface().toLocal(ali->displacement());
	 align::RotationType rotation = ali->surface().toLocal(ali->rotation());

	 double mxx = rotation.xx();
	 double myx = rotation.yx();
	 double mzx = rotation.zx();
	 double mzy = rotation.zy();
	 double mzz = rotation.zz();
	 double denom = sqrt(1. - mzx*mzx);
	 
	 m_before_rawid = m_rawid[ali];
	 m_before_level = ali->alignableObjectId();
	 m_before_x = displacement.x();
	 m_before_y = displacement.y();
	 m_before_z = displacement.z();
	 m_before_phix = atan2(-mzy/denom, mzz/denom);
	 m_before_phiy = atan2(mzx, denom);
	 m_before_phiz = atan2(-myx/denom, mxx/denom);
	 if (m_before) m_before->Fill();

	 if (disk != m_disk_index.end()  ||  chamber != m_chamber_index.end()  ||  layer != m_layer_index.end()) {

	    if (iteration() == 1) {
	       if (disk != m_disk_index.end()) {
		  if (m_conv_x[disk->second]) m_conv_x[disk->second]->SetBinContent(1, m_before_x);
		  if (m_conv_y[disk->second]) m_conv_y[disk->second]->SetBinContent(1, m_before_y);
		  if (m_conv_z[disk->second]) m_conv_z[disk->second]->SetBinContent(1, m_before_z);
		  if (m_conv_phix[disk->second]) m_conv_phix[disk->second]->SetBinContent(1, m_before_phix);
		  if (m_conv_phiy[disk->second]) m_conv_phiy[disk->second]->SetBinContent(1, m_before_phiy);
		  if (m_conv_phiz[disk->second]) m_conv_phiz[disk->second]->SetBinContent(1, m_before_phiz);
	       }

	       if (chamber != m_chamber_index.end()) {
		  if (m_conv_x[chamber->second]) m_conv_x[chamber->second]->SetBinContent(1, m_before_x);
		  if (m_conv_y[chamber->second]) m_conv_y[chamber->second]->SetBinContent(1, m_before_y);
		  if (m_conv_z[chamber->second]) m_conv_z[chamber->second]->SetBinContent(1, m_before_z);
		  if (m_conv_phix[chamber->second]) m_conv_phix[chamber->second]->SetBinContent(1, m_before_phix);
		  if (m_conv_phiy[chamber->second]) m_conv_phiy[chamber->second]->SetBinContent(1, m_before_phiy);
		  if (m_conv_phiz[chamber->second]) m_conv_phiz[chamber->second]->SetBinContent(1, m_before_phiz);
	       }

	       if (layer != m_layer_index.end()) {
		  if (m_conv_x[layer->second]) m_conv_x[layer->second]->SetBinContent(1, m_before_x);
		  if (m_conv_y[layer->second]) m_conv_y[layer->second]->SetBinContent(1, m_before_y);
		  if (m_conv_z[layer->second]) m_conv_z[layer->second]->SetBinContent(1, m_before_z);
		  if (m_conv_phix[layer->second]) m_conv_phix[layer->second]->SetBinContent(1, m_before_phix);
		  if (m_conv_phiy[layer->second]) m_conv_phiy[layer->second]->SetBinContent(1, m_before_phiy);
		  if (m_conv_phiz[layer->second]) m_conv_phiz[layer->second]->SetBinContent(1, m_before_phiz);
	       }
	    } // end if this is iteration 1
	 } // end if we have a histogram for this alignable
      } // end ascent to topmost alignable
   } // end loop over alignables

   if (m_createPythonGeometry) createPythonGeometry();
}

void AlignmentMonitorMuonHIP::bookByAli(const char *level, const int rawid, const unsigned int index) {
   bool dt = (DetId(rawid).subdetId() == MuonSubdetId::DT);

   char dir[256], name[256], title[256];

   if (m_book_nhits_vsiter) {
      sprintf(dir, "/nhits_vsiter_%s/", level);
      sprintf(name, "nhits_vsiter_%s_%d", level, rawid);
      sprintf(title, "Number of hits on %s %d vs iteration", level, rawid);
      m_nhits_vsiter[index] = book1D(dir, name, title, m_params_iterations, m_params_iterations_low, m_params_iterations_high);
   }
   else {
      m_nhits_vsiter[index] = NULL;
   }

   if (m_book_conv_x) {
      sprintf(dir, "/conv_x_%s/", level);
      sprintf(name, "conv_x_%s_%d", level, rawid);
      sprintf(title, "Convergence in x of %s %d vs iteration", level, rawid);
      m_conv_x[index] = book1D(dir, name, title, m_params_iterations, m_params_iterations_low, m_params_iterations_high);
   }
   else {
      m_conv_x[index] = NULL;
   }

   if (m_book_conv_y) {
      sprintf(dir, "/conv_y_%s/", level);
      sprintf(name, "conv_y_%s_%d", level, rawid);
      sprintf(title, "Convergence in y of %s %d vs iteration", level, rawid);
      m_conv_y[index] = book1D(dir, name, title, m_params_iterations, m_params_iterations_low, m_params_iterations_high);
   }
   else {
      m_conv_y[index] = NULL;
   }

   if (m_book_conv_z) {
      sprintf(dir, "/conv_z_%s/", level);
      sprintf(name, "conv_z_%s_%d", level, rawid);
      sprintf(title, "Convergence in z of %s %d vs iteration", level, rawid);
      m_conv_z[index] = book1D(dir, name, title, m_params_iterations, m_params_iterations_low, m_params_iterations_high);
   }
   else {
      m_conv_z[index] = NULL;
   }

   if (m_book_conv_phix) {
      sprintf(dir, "/conv_phix_%s/", level);
      sprintf(name, "conv_phix_%s_%d", level, rawid);
      sprintf(title, "Convergence in phix of %s %d vs iteration", level, rawid);
      m_conv_phix[index] = book1D(dir, name, title, m_params_iterations, m_params_iterations_low, m_params_iterations_high);
   }
   else {
      m_conv_phix[index] = NULL;
   }

   if (m_book_conv_phiy) {
      sprintf(dir, "/conv_phiy_%s/", level);
      sprintf(name, "conv_phiy_%s_%d", level, rawid);
      sprintf(title, "Convergence in phiy of %s %d vs iteration", level, rawid);
      m_conv_phiy[index] = book1D(dir, name, title, m_params_iterations, m_params_iterations_low, m_params_iterations_high);
   }
   else {
      m_conv_phiy[index] = NULL;
   }

   if (m_book_conv_phiz) {
      sprintf(dir, "/conv_phiz_%s/", level);
      sprintf(name, "conv_phiz_%s_%d", level, rawid);
      sprintf(title, "Convergence in phiz of %s %d vs iteration", level, rawid);
      m_conv_phiz[index] = book1D(dir, name, title, m_params_iterations, m_params_iterations_low, m_params_iterations_high);
   }
   else {
      m_conv_phiz[index] = NULL;
   }

   if (m_book_xresid) {
      sprintf(dir, "/iterN/xresid_%s/", level);
      sprintf(name, "xresid_%s_%d", level, rawid);
      sprintf(title, "x residual on %s %d for iteration %d", level, rawid, iteration());
      m_xresid[index] = book1D(dir, name, title, m_params_bins, m_params_xresid_low, m_params_xresid_high);
   }
   else {
      m_xresid[index] = NULL;
   }
   
   if (m_book_xresidwide) {
      sprintf(dir, "/iterN/xresidwide_%s/", level);
      sprintf(name, "xresidwide_%s_%d", level, rawid);
      sprintf(title, "x residual on %s %d for iteration %d", level, rawid, iteration());
      m_xresidwide[index] = book1D(dir, name, title, m_params_bins, m_params_xresidwide_low, m_params_xresidwide_high);
   }
   else {
      m_xresidwide[index] = NULL;
   }

   if (m_book_yresid  &&  !dt) {
      sprintf(dir, "/iterN/yresid_%s/", level);
      sprintf(name, "yresid_%s_%d", level, rawid);
      sprintf(title, "y residual on %s %d for iteration %d", level, rawid, iteration());
      m_yresid[index] = book1D(dir, name, title, m_params_bins, m_params_yresid_low, m_params_yresid_high);
   }
   else {
      m_yresid[index] = NULL;
   }

   if (m_book_wxresid) {
      sprintf(dir, "/iterN/wxresid_%s/", level);
      sprintf(name, "wxresid_%s_%d", level, rawid);
      sprintf(title, "Weighted x residual on %s %d for iteration %d", level, rawid, iteration());
      m_wxresid[index] = book1D(dir, name, title, m_params_bins, m_params_xresid_low, m_params_xresid_high);
   }
   else {
      m_wxresid[index] = NULL;
   }

   if (m_book_wxresidwide) {
      sprintf(dir, "/iterN/wxresidwide_%s/", level);
      sprintf(name, "wxresidwide_%s_%d", level, rawid);
      sprintf(title, "Weighted x residual on %s %d for iteration %d", level, rawid, iteration());
      m_wxresidwide[index] = book1D(dir, name, title, m_params_bins, m_params_xresidwide_low, m_params_xresidwide_high);
   }
   else {
      m_wxresidwide[index] = NULL;
   }

   if (m_book_wyresid  &&  !dt) {
      sprintf(dir, "/iterN/wyresid_%s/", level);
      sprintf(name, "wyresid_%s_%d", level, rawid);
      sprintf(title, "Weighted y residual on %s %d for iteration %d", level, rawid, iteration());
      m_wyresid[index] = book1D(dir, name, title, m_params_bins, m_params_yresid_low, m_params_yresid_high);
   }
   else {
      m_wyresid[index] = NULL;
   }

   if (m_book_wxresid_vsx) {
      sprintf(dir, "/iterN/wxresid_vsx_%s/", level);
      sprintf(name, "wxresid_vsx_%s_%d", level, rawid);
      sprintf(title, "Weighted x residual vs track's x on %s %d for iteration %d", level, rawid, iteration());
      m_wxresid_vsx[index] = bookProfile(dir, name, title, m_params_bins, (dt? m_params_xDT_low: m_params_xCSC_low), (dt? m_params_xDT_high: m_params_xCSC_high), 1, m_params_xresidwide_low, m_params_xresidwide_high);
   }
   else {
      m_wxresid_vsx[index] = NULL;
   }

   if (m_book_wxresid_vsy) {
      sprintf(dir, "/iterN/wxresid_vsy_%s/", level);
      sprintf(name, "wxresid_vsy_%s_%d", level, rawid);
      sprintf(title, "Weighted x residual vs track's y on %s %d for iteration %d", level, rawid, iteration());
      m_wxresid_vsy[index] = bookProfile(dir, name, title, m_params_bins, (dt? m_params_yDT_low: m_params_yCSC_low), (dt? m_params_yDT_high: m_params_yCSC_high), 1, m_params_xresidwide_low, m_params_xresidwide_high);
   }
   else {
      m_wxresid_vsy[index] = NULL;
   }

   if (m_book_wyresid_vsx  &&  !dt) {
      sprintf(dir, "/iterN/wyresid_vsx_%s/", level);
      sprintf(name, "wyresid_vsx_%s_%d", level, rawid);
      sprintf(title, "Weighted y residual vs track's x on %s %d for iteration %d", level, rawid, iteration());
      m_wyresid_vsx[index] = bookProfile(dir, name, title, m_params_bins, (dt? m_params_xDT_low: m_params_xCSC_low), (dt? m_params_xDT_high: m_params_xCSC_high), 1, m_params_yresid_low, m_params_yresid_high);
   }
   else {
      m_wyresid_vsx[index] = NULL;
   }

   if (m_book_wyresid_vsy  &&  !dt) {
      sprintf(dir, "/iterN/wyresid_vsy_%s/", level);
      sprintf(name, "wyresid_vsy_%s_%d", level, rawid);
      sprintf(title, "Weighted y residual vs track's y on %s %d for iteration %d", level, rawid, iteration());
      m_wyresid_vsy[index] = bookProfile(dir, name, title, m_params_bins, (dt? m_params_yDT_low: m_params_yCSC_low), (dt? m_params_yDT_high: m_params_yCSC_high), 1, m_params_yresid_low, m_params_yresid_high);
   }
   else {
      m_wyresid_vsy[index] = NULL;
   }

   if (m_book_xpull) {
      sprintf(dir, "/iterN/xpull_%s/", level);
      sprintf(name, "xpull_%s_%d", level, rawid);
      sprintf(title, "x pull distribution on %s %d for iteration %d", level, rawid, iteration());
      m_xpull[index] = book1D(dir, name, title, m_params_bins, m_params_xpull_low, m_params_xpull_high);
   }
   else {
      m_xpull[index] = NULL;
   }

   if (m_book_ypull  &&  !dt) {
      sprintf(dir, "/iterN/ypull_%s/", level);
      sprintf(name, "ypull_%s_%d", level, rawid);
      sprintf(title, "y pull distribution on %s %d for iteration %d", level, rawid, iteration());
      m_ypull[index] = book1D(dir, name, title, m_params_bins, m_params_ypull_low, m_params_ypull_high);
   }
   else {
      m_ypull[index] = NULL;
   }
}

//////////////////////////////////////////////////////////////////////
// event()
//////////////////////////////////////////////////////////////////////

void AlignmentMonitorMuonHIP::event(const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& tracks) {
   TrajectoryStateCombiner tsoscomb;

   for (ConstTrajTrackPairCollection::const_iterator it = tracks.begin();  it != tracks.end();  ++it) {
      const Trajectory *traj = it->first;
//      const reco::Track *track = it->second;

      std::vector<TrajectoryMeasurement> measurements = traj->measurements();
      for (std::vector<TrajectoryMeasurement>::const_iterator im = measurements.begin();  im != measurements.end();  ++im) {
	 const TrajectoryMeasurement meas = *im;
	 const TransientTrackingRecHit* hit = &(*meas.recHit());
	 const DetId id = hit->geographicalId();

	 if (hit->isValid()  &&  pNavigator()->detAndSubdetInMap(id)) {
	    Alignable *alignable = pNavigator()->alignableFromDetId(id);
	    bool y_valid = (alignable->geomDetId().subdetId() == MuonSubdetId::CSC);

	    std::map<Alignable*, unsigned int>::const_iterator disk = m_disk_index.find(alignable);
	    std::map<Alignable*, unsigned int>::const_iterator chamber = m_chamber_index.find(alignable);
	    std::map<Alignable*, unsigned int>::const_iterator layer = m_layer_index.find(alignable);

	    Alignable *ascend = alignable;
	    while (disk == m_disk_index.end()  &&  (ascend = ascend->mother())) disk = m_disk_index.find(ascend);
	    ascend = alignable;
	    while (chamber == m_chamber_index.end()  &&  (ascend = ascend->mother())) chamber = m_chamber_index.find(ascend);
	    ascend = alignable;
	    while (layer == m_layer_index.end()  &&  (ascend = ascend->mother())) layer = m_layer_index.find(ascend);
	       
	    TrajectoryStateOnSurface tsosc = tsoscomb.combine(meas.forwardPredictedState(), meas.backwardPredictedState());
	    LocalPoint trackPos = tsosc.localPosition();
	    LocalError trackErr = tsosc.localError().positionError();
	    LocalPoint hitPos = hit->localPosition();
	    LocalError hitErr = hit->localPositionError();

	    double x_residual = trackPos.x() - hitPos.x();
	    double y_residual = trackPos.y() - hitPos.y();
	    double x_reserr2 = trackErr.xx() + hitErr.xx();
	    double y_reserr2 = trackErr.yy() + hitErr.yy();
	    double xpos = trackPos.x();
	    double ypos = trackPos.y();

	    if (disk != m_disk_index.end()  ||  chamber != m_chamber_index.end()  ||  layer != m_layer_index.end()) {

	       if (disk != m_disk_index.end()) {
		  fill(disk->second, x_residual, y_residual, x_reserr2, y_reserr2, xpos, ypos, y_valid);
	       }
	       if (chamber != m_chamber_index.end()) {
		  fill(chamber->second, x_residual, y_residual, x_reserr2, y_reserr2, xpos, ypos, y_valid);
	       }
	       if (layer != m_layer_index.end()) {
		  fill(layer->second, x_residual, y_residual, x_reserr2, y_reserr2, xpos, ypos, y_valid);
	       }

	    } // end if we're plotting this hit

	    fill(0, x_residual, y_residual, x_reserr2, y_reserr2, xpos, ypos, y_valid);

	 } // end if hit is valid
      } // end loop over measurements
   } // end loop over tracks
}

void AlignmentMonitorMuonHIP::fill(unsigned int index, double x_residual, double y_residual, double x_reserr2, double y_reserr2, double xpos, double ypos, bool y_valid) {
   if (m_nhits_vsiter[index]) m_nhits_vsiter[index]->Fill(iteration());
		  
   if (m_xresid[index])                                        m_xresid[index]->Fill(x_residual);
   if (m_xresidwide[index])                                    m_xresidwide[index]->Fill(x_residual);
   if (y_valid && m_yresid[index])                             m_yresid[index]->Fill(y_residual);
   if (m_wxresid[index]  &&  x_reserr2 != 0.)                  m_wxresid[index]->Fill(x_residual, 1./x_reserr2);
   if (m_wxresidwide[index]  &&  x_reserr2 != 0.)              m_wxresidwide[index]->Fill(x_residual, 1./x_reserr2);
   if (y_valid && m_wyresid[index]  &&  y_reserr2 != 0.)       m_wyresid[index]->Fill(y_residual, 1./y_reserr2);
   if (m_wxresid_vsx[index]  &&  x_reserr2 != 0.)              m_wxresid_vsx[index]->Fill(xpos, x_residual, 1./x_reserr2);
   if (m_wxresid_vsy[index]  &&  x_reserr2 != 0.)              m_wxresid_vsy[index]->Fill(ypos, x_residual, 1./x_reserr2);
   if (y_valid && m_wyresid_vsx[index]  &&  y_reserr2 != 0.)   m_wyresid_vsx[index]->Fill(xpos, y_residual, 1./y_reserr2);
   if (y_valid && m_wyresid_vsy[index]  &&  y_reserr2 != 0.)   m_wyresid_vsy[index]->Fill(ypos, y_residual, 1./y_reserr2);
   if (m_xpull[index]  &&  x_reserr2 != 0.)                    m_xpull[index]->Fill(x_residual / sqrt(fabs(x_reserr2)));
   if (y_valid && m_ypull[index]  &&  y_reserr2 != 0.)         m_ypull[index]->Fill(y_residual / sqrt(fabs(y_reserr2)));
}

//////////////////////////////////////////////////////////////////////
// afterAlignment()
//////////////////////////////////////////////////////////////////////

void AlignmentMonitorMuonHIP::afterAlignment(const edm::EventSetup &iSetup) {
   std::vector<Alignable*> alignables = pStore()->alignables();
   for (std::vector<Alignable*>::const_iterator aliiter = alignables.begin();  aliiter != alignables.end();  ++aliiter) {
      Alignable *ali = *aliiter;
      std::map<Alignable*, unsigned int>::const_iterator disk = m_disk_index.find(ali);
      std::map<Alignable*, unsigned int>::const_iterator chamber = m_chamber_index.find(ali);
      std::map<Alignable*, unsigned int>::const_iterator layer = m_layer_index.find(ali);
      if (disk != m_disk_index.end()  ||  chamber != m_chamber_index.end()  ||  layer != m_layer_index.end()) {
	 // The central values of all the alignment positions
	 LocalVector displacement = ali->surface().toLocal(ali->displacement());
	 align::RotationType rotation = ali->surface().toLocal(ali->rotation());

	 //////// Obtain the position errors by recalculation //////////////////////////////////////////
	 AlignmentParameters *par = ali->alignmentParameters();
	 HIPUserVariables *uservar = dynamic_cast<HIPUserVariables*>(par->userVariables());
	 AlgebraicSymMatrix jtvj = uservar->jtvj;
	 AlgebraicVector jtve = uservar->jtve;
	 int npar = jtve.num_row();

	 int ierr;
	 AlgebraicSymMatrix jtvjinv = jtvj.inverse(ierr);
	 AlgebraicVector paramerr(npar);
	 if (ierr == 0) {
	    AlgebraicVector params = -(jtvjinv * jtve);
	    for (int i = 0;  i < npar;  i++) {
	       if (fabs(jtvjinv[i][i]) > 0) paramerr[i] = sqrt(fabs(jtvjinv[i][i]));
	       else paramerr[i] = params[i];
	    }
	 }
	 else {
	    for (int i = 0;  i < npar;  i++) {
	       paramerr[i] = 0.;
	    }
	 }

	 std::vector<bool> selector = par->selector();
	 AlgebraicVector allparamerr(6);
	 int j = 0;
	 for (int i = 0;  i < 6;  i++) {
	    if (selector[i]) {
	       allparamerr[i] = paramerr[j];
	       j++;
	    }
	    else {
	       allparamerr[i] = 0.;
	    }
	 }

	 /////// end calculate parameter errors ///////////////////////////////////////////////////////

	 double mxx = rotation.xx();
	 double myx = rotation.yx();
	 double mzx = rotation.zx();
	 double mzy = rotation.zy();
	 double mzz = rotation.zz();
	 double denom = sqrt(1. - mzx*mzx);
	 
	 m_after_rawid = m_rawid[ali];
	 m_after_level = ali->alignableObjectId();
	 m_after_x = displacement.x();
	 m_after_y = displacement.y();
	 m_after_z = displacement.z();
	 m_after_phix = atan2(-mzy/denom, mzz/denom);
	 m_after_phiy = atan2(mzx, denom);
	 m_after_phiz = atan2(-myx/denom, mxx/denom);
	 m_after_xerr = allparamerr[0];
	 m_after_yerr = allparamerr[1];
	 m_after_zerr = allparamerr[2];
	 m_after_phixerr = allparamerr[3];
	 m_after_phiyerr = allparamerr[4];
	 m_after_phizerr = allparamerr[5];
	 if (m_after) m_after->Fill();

	 std::map<Alignable*, unsigned int>::const_iterator disk = m_disk_index.find(ali);
	 std::map<Alignable*, unsigned int>::const_iterator chamber = m_chamber_index.find(ali);
	 std::map<Alignable*, unsigned int>::const_iterator layer = m_layer_index.find(ali);
	 
	 if (disk != m_disk_index.end()) {
	    if (m_conv_x[disk->second]) m_conv_x[disk->second]->SetBinContent(iteration() + 1, m_after_x);
	    if (m_conv_y[disk->second]) m_conv_y[disk->second]->SetBinContent(iteration() + 1, m_after_y);
	    if (m_conv_z[disk->second]) m_conv_z[disk->second]->SetBinContent(iteration() + 1, m_after_z);
	    if (m_conv_phix[disk->second]) m_conv_phix[disk->second]->SetBinContent(iteration() + 1, m_after_phix);
	    if (m_conv_phiy[disk->second]) m_conv_phiy[disk->second]->SetBinContent(iteration() + 1, m_after_phiy);
	    if (m_conv_phiz[disk->second]) m_conv_phiz[disk->second]->SetBinContent(iteration() + 1, m_after_phiz);

	    if (m_conv_x[disk->second]) m_conv_x[disk->second]->SetBinError(iteration() + 1, m_after_xerr);
	    if (m_conv_y[disk->second]) m_conv_y[disk->second]->SetBinError(iteration() + 1, m_after_yerr);
	    if (m_conv_z[disk->second]) m_conv_z[disk->second]->SetBinError(iteration() + 1, m_after_zerr);
	    if (m_conv_phix[disk->second]) m_conv_phix[disk->second]->SetBinError(iteration() + 1, m_after_phixerr);
	    if (m_conv_phiy[disk->second]) m_conv_phiy[disk->second]->SetBinError(iteration() + 1, m_after_phiyerr);
	    if (m_conv_phiz[disk->second]) m_conv_phiz[disk->second]->SetBinError(iteration() + 1, m_after_phizerr);
	 }

	 if (chamber != m_chamber_index.end()) {
	    if (m_conv_x[chamber->second]) m_conv_x[chamber->second]->SetBinContent(iteration() + 1, m_after_x);
	    if (m_conv_y[chamber->second]) m_conv_y[chamber->second]->SetBinContent(iteration() + 1, m_after_y);
	    if (m_conv_z[chamber->second]) m_conv_z[chamber->second]->SetBinContent(iteration() + 1, m_after_z);
	    if (m_conv_phix[chamber->second]) m_conv_phix[chamber->second]->SetBinContent(iteration() + 1, m_after_phix);
	    if (m_conv_phiy[chamber->second]) m_conv_phiy[chamber->second]->SetBinContent(iteration() + 1, m_after_phiy);
	    if (m_conv_phiz[chamber->second]) m_conv_phiz[chamber->second]->SetBinContent(iteration() + 1, m_after_phiz);

	    if (m_conv_x[chamber->second]) m_conv_x[chamber->second]->SetBinError(iteration() + 1, m_after_xerr);
	    if (m_conv_y[chamber->second]) m_conv_y[chamber->second]->SetBinError(iteration() + 1, m_after_yerr);
	    if (m_conv_z[chamber->second]) m_conv_z[chamber->second]->SetBinError(iteration() + 1, m_after_zerr);
	    if (m_conv_phix[chamber->second]) m_conv_phix[chamber->second]->SetBinError(iteration() + 1, m_after_phixerr);
	    if (m_conv_phiy[chamber->second]) m_conv_phiy[chamber->second]->SetBinError(iteration() + 1, m_after_phiyerr);
	    if (m_conv_phiz[chamber->second]) m_conv_phiz[chamber->second]->SetBinError(iteration() + 1, m_after_phizerr);
	 }

	 if (layer != m_layer_index.end()) {
	    if (m_conv_x[layer->second]) m_conv_x[layer->second]->SetBinContent(iteration() + 1, m_after_x);
	    if (m_conv_y[layer->second]) m_conv_y[layer->second]->SetBinContent(iteration() + 1, m_after_y);
	    if (m_conv_z[layer->second]) m_conv_z[layer->second]->SetBinContent(iteration() + 1, m_after_z);
	    if (m_conv_phix[layer->second]) m_conv_phix[layer->second]->SetBinContent(iteration() + 1, m_after_phix);
	    if (m_conv_phiy[layer->second]) m_conv_phiy[layer->second]->SetBinContent(iteration() + 1, m_after_phiy);
	    if (m_conv_phiz[layer->second]) m_conv_phiz[layer->second]->SetBinContent(iteration() + 1, m_after_phiz);

	    if (m_conv_x[layer->second]) m_conv_x[layer->second]->SetBinError(iteration() + 1, m_after_xerr);
	    if (m_conv_y[layer->second]) m_conv_y[layer->second]->SetBinError(iteration() + 1, m_after_yerr);
	    if (m_conv_z[layer->second]) m_conv_z[layer->second]->SetBinError(iteration() + 1, m_after_zerr);
	    if (m_conv_phix[layer->second]) m_conv_phix[layer->second]->SetBinError(iteration() + 1, m_after_phixerr);
	    if (m_conv_phiy[layer->second]) m_conv_phiy[layer->second]->SetBinError(iteration() + 1, m_after_phiyerr);
	    if (m_conv_phiz[layer->second]) m_conv_phiz[layer->second]->SetBinError(iteration() + 1, m_after_phizerr);
	 }

      } // end if this is a DT or CSC
   } // end loop over alignables

   delete [] m_nhits_vsiter;
   delete [] m_conv_x;
   delete [] m_conv_y;
   delete [] m_conv_z;
   delete [] m_conv_phix;
   delete [] m_conv_phiy;
   delete [] m_conv_phiz;
   delete [] m_xresid;
   delete [] m_xresidwide;
   delete [] m_yresid;
   delete [] m_wxresid;
   delete [] m_wxresidwide;
   delete [] m_wyresid;
   delete [] m_wxresid_vsx;
   delete [] m_wxresid_vsy;
   delete [] m_wyresid_vsx;
   delete [] m_wyresid_vsy;
   delete [] m_xpull;
   delete [] m_ypull;

   // The histograms themselves are deleted by the base class.
}

void AlignmentMonitorMuonHIP::createPythonGeometry() {
   edm::LogInfo("AlignmentMonitorMuonHIP") << "Creating Python geometry." << std::endl;
   ofstream python("muonGeometryData.py");

   for (std::map<Alignable*, unsigned int>::const_iterator disk = m_disk_index.begin();  disk != m_disk_index.end();  ++disk) {
      Alignable *ali = disk->first;
      DetId id = DetId(m_rawid[ali]);

      python << "disks[" << id.rawId() << "] = ";
      if (id.subdetId() == MuonSubdetId::DT) {
	 DTChamberId chamberId(id.rawId());
	 python << "Disk(" << id.rawId() << ", 0).setDT(" << chamberId.wheel() << ")";
      }
      else if (id.subdetId() == MuonSubdetId::CSC) {
	 CSCDetId chamberId(id.rawId());
	 python << "Disk(" << id.rawId() << ", " << (chamberId.endcap() == 1? 1: -1) << ").setCSC(" << (chamberId.endcap() == 1? 1: -1)*chamberId.station() << ")";
      }

      GlobalPoint location = ali->surface().toGlobal(LocalPoint(0., 0., 0.));
      GlobalVector xhat = ali->surface().toGlobal(LocalVector(1., 0., 0.));
      GlobalVector yhat = ali->surface().toGlobal(LocalVector(0., 1., 0.));
      GlobalVector zhat = ali->surface().toGlobal(LocalVector(0., 0., 1.));
      python << ".setLoc(" << location.x() << ", " << location.y() << ", " << location.z() << ")";
      python << ".setXhat(" << xhat.x() << ", " << xhat.y() << ", " << xhat.z() << ")";
      python << ".setYhat(" << yhat.x() << ", " << yhat.y() << ", " << yhat.z() << ")";
      python << ".setZhat(" << zhat.x() << ", " << zhat.y() << ", " << zhat.z() << ")" << std::endl;
   }

   for (std::map<Alignable*, unsigned int>::const_iterator chamber = m_chamber_index.begin();  chamber != m_chamber_index.end();  ++chamber) {
      Alignable *ali = chamber->first;
      DetId id = DetId(m_rawid[ali]);

      python << "chambers[" << id.rawId() << "] = ";
      if (id.subdetId() == MuonSubdetId::DT) {
	 DTChamberId chamberId(id.rawId());
	 python << "Chamber(" << id.rawId() << ", 0).setDT(" << chamberId.wheel() << ", " << chamberId.station() << ", " << chamberId.sector() << ")";
      }
      else if (id.subdetId() == MuonSubdetId::CSC) {
	 CSCDetId chamberId(id.rawId());
	 python << "Chamber(" << id.rawId() << ", " << (chamberId.endcap() == 1? 1: -1) << ").setCSC(" << (chamberId.endcap() == 1? 1: -1)*chamberId.station() << ", " << chamberId.ring() << ", " << chamberId.chamber() << ")";
      }

      GlobalPoint location = ali->surface().toGlobal(LocalPoint(0., 0., 0.));
      GlobalVector xhat = ali->surface().toGlobal(LocalVector(1., 0., 0.));
      GlobalVector yhat = ali->surface().toGlobal(LocalVector(0., 1., 0.));
      GlobalVector zhat = ali->surface().toGlobal(LocalVector(0., 0., 1.));
      python << ".setLoc(" << location.x() << ", " << location.y() << ", " << location.z() << ")";
      python << ".setXhat(" << xhat.x() << ", " << xhat.y() << ", " << xhat.z() << ")";
      python << ".setYhat(" << yhat.x() << ", " << yhat.y() << ", " << yhat.z() << ")";
      python << ".setZhat(" << zhat.x() << ", " << zhat.y() << ", " << zhat.z() << ")" << std::endl;
   }

   for (std::map<Alignable*, unsigned int>::const_iterator layer = m_layer_index.begin();  layer != m_layer_index.end();  ++layer) {
      Alignable *ali = layer->first;
      DetId id = DetId(m_rawid[ali]);

      python << "layers[" << id.rawId() << "] = ";
      if (id.subdetId() == MuonSubdetId::DT) {
	 DTSuperLayerId layerId(id.rawId());
	 python << "Layer(" << id.rawId() << ", 0).setDT(" << layerId.wheel() << ", " << layerId.station() << ", " << layerId.sector() << ", " << layerId.superLayer() << ")";
      }
      else if (id.subdetId() == MuonSubdetId::CSC) {
	 CSCDetId layerId(id.rawId());
	 python << "Layer(" << id.rawId() << ", " << (layerId.endcap() == 1? 1: -1) << ").setCSC(" << (layerId.endcap() == 1? 1: -1)*layerId.station() << ", " << layerId.ring() << ", " << layerId.chamber() << ", " << layerId.layer() << ")";
      }

      GlobalPoint location = ali->surface().toGlobal(LocalPoint(0., 0., 0.));
      GlobalVector xhat = ali->surface().toGlobal(LocalVector(1., 0., 0.));
      GlobalVector yhat = ali->surface().toGlobal(LocalVector(0., 1., 0.));
      GlobalVector zhat = ali->surface().toGlobal(LocalVector(0., 0., 1.));
      python << ".setLoc(" << location.x() << ", " << location.y() << ", " << location.z() << ")";
      python << ".setXhat(" << xhat.x() << ", " << xhat.y() << ", " << xhat.z() << ")";
      python << ".setYhat(" << yhat.x() << ", " << yhat.y() << ", " << yhat.z() << ")";
      python << ".setZhat(" << zhat.x() << ", " << zhat.y() << ", " << zhat.z() << ")" << std::endl;
   }

   edm::LogInfo("AlignmentMonitorMuonHIP") << "Done with Python geometry!" << std::endl;
}

//
// const member functions
//

//
// static member functions
//

//
// SEAL definitions
//

DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorMuonHIP, "AlignmentMonitorMuonHIP");
