// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDisplayEvent
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Mon Dec  3 08:38:38 PST 2007
// $Id: FWDisplayEvent.cc,v 1.11 2008/01/07 05:48:46 chrjones Exp $
//

// system include files
#include <sstream>
#include "TEveManager.h"
#include "TEveViewer.h"
#include "TEveBrowser.h"
#include "TEveGeoNode.h"
#include "TSystem.h"
#include "TEveProjectionManager.h"
#include "TEveScene.h"
#include "TGLViewer.h"
#include "TClass.h"

//geometry
#include "TFile.h"
#include "TEveGeoShapeExtract.h"
#include "TROOT.h"

#include "TGButton.h"

//needed to work around a bug
#include "TApplication.h"

// user include files
#include "Fireworks/Core/interface/FWDisplayEvent.h"
#include "Fireworks/Core/interface/FWRhoPhiZViewManager.h"
#include "Fireworks/Core/interface/FW3DLegoViewManager.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "THStack.h"
#include "TCanvas.h"
#include "TVirtualHistPainter.h"
#include "TH2F.h"
#include "TView.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWDisplayEvent::FWDisplayEvent() :
  m_continueProcessingEvents(false),
  m_waitForUserAction(true),
  m_code(0)
  //m_geom(0),
  //m_rhoPhiProjMgr(0),
  //m_legoCanvas(0)

{
  const char* cmspath = gSystem->Getenv("CMSSW_BASE");
  if(0 == cmspath) {
    throw std::runtime_error("CMSSW_BASE environment variable not set");
  }
  //tell ROOT where to find our macros
  std::string macPath(cmspath);
  macPath += "/src/Fireworks/Core/macros";
  gROOT->SetMacroPath(macPath.c_str());  

/*
   //will eventually get this info from the configuration
  m_physicsTypes.push_back("Tracks");
  m_physicsElements.push_back(0);
  m_physicsProxyBuilderNames.push_back("TracksProxy3DBuilder");

  // register muons
  m_physicsTypes.push_back("Muons");
  m_physicsElements.push_back(0);
  m_physicsProxyBuilderNames.push_back("MuonsProxy3DBuilder");
*/

  // These are only needed temporarilty to work around a problem which 
  // Matevz has patched in a later version of the code
  TApplication::NeedGraphicsLibs();
  gApplication->InitializeGraphics();

  TEveManager::Create();
  TEveBrowser* browser = gEve->GetBrowser();
  //should check to see if already has our tab
  {
    browser->StartEmbedding(TRootBrowser::kLeft);
    {
      TGMainFrame* frmMain=new TGMainFrame(gClient->GetRoot(),
					   1000,
					   600);
      frmMain->SetWindowName("GUI");
      frmMain->SetCleanup(kDeepCleanup);

      TGHorizontalFrame* hf = new TGHorizontalFrame(frmMain);
      //We need an error handling system which can properly report
      // errors and decide what to do
      // given that we are an interactive system we need to leave
      // the code in a good state so that users can decided to 
      // continue or not
      {
	if(0==gSystem->Getenv("ROOTSYS")) {
	  std::cerr<<"environment variable ROOTSYS is not set" <<
	    std::endl;
	  throw std::runtime_error("ROOTSYS environment variable not set");
	}
	TString icondir(Form("%s/icons/",gSystem->Getenv("ROOTSYS")));
	
	m_homeButton= new TGPictureButton(hf, gClient->GetPicture(icondir+"GoHome.gif"));
	hf->AddFrame(m_homeButton);
	m_homeButton->Connect("Clicked()", "FWDisplayEvent", this, "goHome()");
	 
	m_advanceButton= new TGPictureButton(hf, gClient->GetPicture(icondir+"GoForward.gif"));
	hf->AddFrame(m_advanceButton);
	m_advanceButton->Connect("Clicked()", "FWDisplayEvent", this, "goForward()");
	
	m_backwardButton= new TGPictureButton(hf, gClient->GetPicture(icondir+"GoBack.gif"));
	hf->AddFrame(m_backwardButton);
	m_backwardButton->Connect("Clicked()", "FWDisplayEvent", this, "goBack()");
	 
	m_stopButton= new TGPictureButton(hf, gClient->GetPicture(icondir+"StopLoading.gif"));
	hf->AddFrame(m_stopButton);
	m_stopButton->Connect("Clicked()", "FWDisplayEvent", this, "stop()");
	
      }
      frmMain->AddFrame(hf);

      frmMain->MapSubwindows();
      frmMain->Resize();
      frmMain->MapWindow();
    }
    browser->StopEmbedding();
    browser->SetTabTitle("Event Control",0);
  }

  boost::shared_ptr<FWViewManagerBase> rpzViewManager(
		  new FWRhoPhiZViewManager());
  m_viewManagers.push_back(rpzViewManager);
  m_viewManagers.push_back( boost::shared_ptr<FWViewManagerBase>(
			       new FW3DLegoViewManager()));
  /*
  //setup projection
  TEveViewer* nv = gEve->SpawnNewViewer("Rho Phi");
  nv->GetGLViewer()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
  TEveScene* ns = gEve->SpawnNewScene("Rho Phi");
  nv->AddScene(ns);

  m_rhoPhiProjMgr = new TEveProjectionManager;
  gEve->AddToListTree(m_rhoPhiProjMgr,kTRUE);
  gEve->AddElement(m_rhoPhiProjMgr,ns);
  */
  //handle geometry
  /*
  gGeoManager = gEve->GetGeometry("cmsGeom20.root");

  TEveElementList* gL = 
    new TEveElementList("CMS");
  gEve->AddGlobalElement(gL);

  TGeoNode* node = gGeoManager->GetTopVolume()->GetNode(0);
  TEveGeoTopNode* re = new TEveGeoTopNode(gGeoManager,
					  node);
  re->UseNodeTrans();
  gEve->AddGlobalElement(re,gL);
  */
   
   // FIXME: something is wrong with losing geomtry when file is closed or 
   // some other files are opened.
   /*
   TFile* f = TFile::Open("tracker.root");
  if(not f->IsOpen()) {
    std::cerr <<"failed to open 'tracker.root'"<<std::endl;
    throw std::runtime_error("Failed to open 'tracker.root' geometry file");
  }
  TEveGeoShapeExtract* gse = dynamic_cast<TEveGeoShapeExtract*>(f->Get("Tracker"));
  TEveGeoShape* gsre = TEveGeoShape::ImportShapeExtract(gse,0);
  f->Close();
  m_geom = gsre;
    */
  //kTRUE tells it to reset the camera so we see everything 
  //gEve->Redraw3D(kTRUE);  

  //m_legoCanvas = gEve->AddCanvasTab("legoCanvas");
   // one way of connecting event processing function to a canvas
   // m_legoCanvas->AddExec("ex", "FWDisplayEvent::DynamicCoordinates()");

   // use Qt messaging mechanism
   //m_legoCanvas->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)","FWDisplayEvent", this,
  //			 "exec3event(Int_t,Int_t,Int_t,TObject*)");

   gSystem->ProcessEvents();
}

// FWDisplayEvent::FWDisplayEvent(const FWDisplayEvent& rhs)
// {
//    // do actual copying here;
// }

FWDisplayEvent::~FWDisplayEvent()
{
}

//
// assignment operators
//
// const FWDisplayEvent& FWDisplayEvent::operator=(const FWDisplayEvent& rhs)
// {
//   //An exception safe implementation is
//   FWDisplayEvent temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void FWDisplayEvent::registerEventItem(const FWEventItem&iItem)
{
  const FWEventItem* newItem = m_eiManager.add(iItem);
  for(std::vector<boost::shared_ptr<FWViewManagerBase> >::iterator itVM = m_viewManagers.begin();
      itVM != m_viewManagers.end();
      ++itVM) {
    (*itVM)->newItem(newItem);
  }
}

void FWDisplayEvent::registerProxyBuilder(const std::string& type, 
					  const std::string& proxyBuilderName)
{
  for(std::vector<boost::shared_ptr<FWViewManagerBase> >::iterator itVM = m_viewManagers.begin();
      itVM != m_viewManagers.end();
      ++itVM) {
    if((*itVM)->useableBuilder(proxyBuilderName)) {
      std::cout <<"REGISTERING "<<type<<std::endl;
      (*itVM)->registerProxyBuilder(type,proxyBuilderName);
      break;
    }
  }


   //Do we have the necessary EventItem?
  /*
   const FWEventItem* item = m_eiManager.find(type);
   if(0 == item) {
     //Need to give error message here!
     return;
   }
   
   FWModelProxy proxy;
   proxy.type = type;
   proxy.builderName = proxyBuilderName;
   
   //create proxy builders
   Int_t error;
   TClass *baseClass = TClass::GetClass(typeid(FWDataProxyBuilder));
   assert(baseClass !=0);

   //does the class already exist?
   TClass *c = TClass::GetClass( proxy.builderName.c_str() );
   if(0==c) {
      //try to load a macro of that name
      
      //How can I tell if this succeeds or failes? error and value are always 0!
      // I could not make the non-compiled mechanism to work without seg-faults
      Int_t value = gROOT->LoadMacro( (proxy.builderName+".C+").c_str(), &error );
      c = TClass::GetClass( proxy.builderName.c_str() );
      if(0==c ) {
	 std::cerr <<"failed to find "<< proxy.builderName << std::endl;
	 return;
      }
   }
   FWDataProxyBuilder* builder = 0;
   void* inst = c->New();
   builder = reinterpret_cast<FWDataProxyBuilder*>(c->DynamicCast(baseClass,inst));
   if(0==builder) {
      std::cerr<<"conversion to FWDataProxyBuilder failed"<<std::endl;
      return;
   }
   proxy.builder = boost::shared_ptr<FWDataProxyBuilder>(builder);
   proxy.builder->setItem(item);
   
   m_modelProxies.push_back( proxy );
  */
}

void
FWDisplayEvent::goForward()
{
  m_continueProcessingEvents = true;
  m_code = 1;
}

void
FWDisplayEvent::goBack()
{
  m_continueProcessingEvents = true;
  m_code = -1;
}

void
FWDisplayEvent::goHome()
{
  m_continueProcessingEvents = true;
  m_code = -2;
}

void
FWDisplayEvent::stop()
{
  m_continueProcessingEvents = true;
  m_code = -3;
}

void
FWDisplayEvent::waitForUserAction()
{
  m_waitForUserAction = true;
}

void
FWDisplayEvent::doNotWaitForUserAction()
{
  m_waitForUserAction = false;
}

//
// const member functions
//
bool
FWDisplayEvent::waitingForUserAction() const
{
  return m_waitForUserAction;
}

namespace {
  //guarantee that no matter how we go back to Cint that
  // we have disabled these buttons
  struct EnableButton {
    EnableButton( TGButton* iButton):
      m_button(iButton)
    {
      if(0!=m_button) {
	m_button->SetEnabled();
      }
    }
    ~EnableButton()
    {
      m_button->SetEnabled(kFALSE);
      gSystem->DispatchOneEvent(kFALSE);
    }

  private:
    TGButton* m_button;
  };

}
int
FWDisplayEvent::draw(const fwlite::Event& iEvent)
{
  const FWDisplayEvent* c = this;
  return c->draw(iEvent);
}

int
FWDisplayEvent::draw(const fwlite::Event& iEvent) const
{
  //need to reset
  m_continueProcessingEvents = false;
  EnableButton homeB(m_homeButton);
  EnableButton advancedB(m_advanceButton);
  EnableButton backwardB(m_backwardButton);
  EnableButton stopB(m_stopButton);

  using namespace std;
  if(0==gEve) {
    cout <<"Eve not initialized"<<endl;
  }
   
  m_eiManager.newEvent(&iEvent);

  for(std::vector<boost::shared_ptr<FWViewManagerBase> >::const_iterator itVM = m_viewManagers.begin();
      itVM != m_viewManagers.end();
      ++itVM) {
    (*itVM)->newEventAvailable();
  }

  /*
  {
    //while inside this scope, do not let
    // Eve do any redrawing
    TEveManager::TRedrawDisabler disableRedraw(gEve);

     // build models
     for ( std::vector<FWModelProxy>::iterator proxy = m_modelProxies.begin();
	   proxy != m_modelProxies.end(); ++proxy )
       proxy->builder->build(&(proxy->product) );
     
     // R-Phi projections
     
     // setup the projection
     // each projection knows what model proxies it needs
     // NOTE: this should be encapsulated and made configurable 
     //       somewhere else.
     m_rhoPhiProjMgr->DestroyElements();
     
	  
     // FIXME - standard way of loading geomtry failed
     // ----------- from here 
     if ( ! m_geom ) {
	TFile f("tracker.root");
	if(not f.IsOpen()) {
	   std::cerr <<"failed to open 'tracker.root'"<<std::endl;
	   throw std::runtime_error("Failed to open 'tracker.root' geometry file");
	}
	TEveGeoShapeExtract* gse = dynamic_cast<TEveGeoShapeExtract*>(f.Get("Tracker"));
	TEveGeoShape* gsre = TEveGeoShape::ImportShapeExtract(gse,0);
	f.Close();
	m_geom = gsre;
     }
     // ---------- to here
     
     m_rhoPhiProjMgr->ImportElements(m_geom);
     for ( std::vector<FWModelProxy>::iterator proxy = m_modelProxies.begin();
	   proxy != m_modelProxies.end(); ++proxy ) 
       {
	  if ( proxy->builderName == "TracksProxy3DBuilder" ||
	       proxy->builderName == "MuonsProxy3DBuilder"  ) 
	    {
	       TEveElementList* list = dynamic_cast<TEveElementList*>(proxy->product);
	       m_rhoPhiProjMgr->ImportElements(list);
	    }
       }
     
     // LEGO
     drawLego();
     //At the end of this scope, redrawing will be enabled
  }
  */
  
  //check for input at least once
  gSystem->ProcessEvents();
  while(not gROOT->IsInterrupted() and
	m_waitForUserAction and 
	not m_continueProcessingEvents) {
     // gSystem->ProcessEvents();
     gSystem->DispatchOneEvent(kFALSE);
  }
  return m_code;
}

/*
void FWDisplayEvent::drawLego() const
{
    // do not let Eve do any redrawing
   TEveManager::TRedrawDisabler disableRedraw(gEve);

   for ( std::vector<FWModelProxy>::iterator proxy = m_modelProxies.begin();
	 proxy != m_modelProxies.end(); ++proxy ) 
     {
	if ( proxy->builderName == "CaloProxyLegoBuilder" ) 
	  {
	     if ( TList* list = dynamic_cast<TList*>(proxy->product) )
	       if ( THStack* stack = dynamic_cast<THStack*>(list->FindObject("LegoStack") ) ) {
		  bool firstDraw = (stack->GetHistogram()==0);
		  m_legoCanvas->cd();
		  stack->Draw("lego1 fb bb");
		        // default parameters
		  if ( firstDraw ) {
		     stack->GetHistogram()->GetXaxis()->SetRangeUser(-1.74,1.74); // zoomed in default view
		     stack->GetHistogram()->GetXaxis()->SetTitle("#eta");
		     stack->GetHistogram()->GetYaxis()->SetTitle("#phi");
		     stack->GetHistogram()->GetXaxis()->SetLabelSize(0.03);
		     stack->GetHistogram()->GetXaxis()->SetTickLength(0.02);
		     stack->GetHistogram()->GetXaxis()->SetTitleOffset(1.2);
		     stack->GetHistogram()->GetYaxis()->SetLabelSize(0.03);
		     stack->GetHistogram()->GetYaxis()->SetTickLength(0.02);
		     stack->GetHistogram()->GetYaxis()->SetTitleOffset(1.2);
		     stack->GetHistogram()->GetZaxis()->SetTitle("Et, [GeV]");
		     stack->GetHistogram()->GetZaxis()->SetLabelSize(0.03);
		     stack->GetHistogram()->GetZaxis()->SetTickLength(0.02); 
		  }
		  stack->Draw("lego1 fb bb");
		  m_legoCanvas->Modified();
		  m_legoCanvas->Update();
	       }
	  }
     }
}
*/

//
// static member functions
//
/*
void FWDisplayEvent::DynamicCoordinates()
{
   // Buttons.h
   int event = gPad->GetEvent();
   if (event != kButton1Down) return;
   std::cout << "Event: " << event << std::endl;
   gPad->GetCanvas()->FeedbackMode(kTRUE);
   
   int px = gPad->GetEventX();
   int py = gPad->GetEventY();
   std::cout << px << "   " << py << std::endl;
}
 
void FWDisplayEvent::exec3event(int event, int x, int y, TObject *selected)
{
   // Two modes of tower selection is supported:
   // - selection based on the base of a tower (point with z=0)
   // - project view of a tower (experimental)
   bool projectedMode = true;
   TCanvas *c = (TCanvas *) gTQSender;
   if (event == kButton2Down) {
      printf("Canvas %s: event=%d, x=%d, y=%d, selected=%s\n", c->GetName(),
	     event, x, y, selected->IsA()->GetName());
      THStack* stack = dynamic_cast<THStack*>(gPad->GetPrimitive("LegoStack"));
      if ( ! stack) return;
      TH2F* ecal = dynamic_cast<TH2F*>(stack->GetHists()->FindObject("ecalLego"));
      TH2F* hcal = dynamic_cast<TH2F*>(stack->GetHists()->FindObject("hcalLego"));
      // TH2F* h = dynamic_cast<TH2F*>(selected);
      if ( ecal && hcal ){
	 double zMax = 0.001;
	 if ( projectedMode ) zMax = stack->GetMaximum();
	 int selectedXbin(0), selectedYbin(0);
	 double selectedX(0), selectedY(0), selectedZ(0), selectedECAL(0), selectedHCAL(0);
	 
	 // scan non-zero z 
	 int oldx(0), oldy(0);
	 for ( double z = 0; z<zMax; z+=1) {
	    Double_t wcX,wcY;
	    pixel2wc(x,y,wcX,wcY,z);
	    int xbin = stack->GetXaxis()->FindFixBin(wcX);
	    int ybin = stack->GetYaxis()->FindFixBin(wcY);
	    if (oldx == xbin && oldy == ybin) continue;
	    oldx = xbin; 
	    oldy = ybin;
	    if ( xbin > stack->GetXaxis()->GetNbins() || ybin > stack->GetYaxis()->GetNbins() ) continue;
	    double content = ecal->GetBinContent(xbin,ybin)+hcal->GetBinContent(xbin,ybin);
	    if ( z <= content ) {
	       selectedXbin = xbin;
	       selectedYbin = ybin;
	       selectedX = wcX;
	       selectedY = wcY;
	       selectedZ = z;
	       selectedECAL = ecal->GetBinContent(xbin,ybin);
	       selectedHCAL = hcal->GetBinContent(xbin,ybin);
	    }
	 }
	 if ( selectedXbin > 0 && selectedYbin>0 )
	   std::cout << "x=" << selectedX << ", y=" << selectedY << ", z=" << selectedZ << 
	   ", xbin=" << selectedXbin << ", ybin=" << selectedYbin << ", Et (total, em, had): " <<  
	   selectedECAL+selectedHCAL << ", " << selectedECAL << ", " << selectedHCAL << std::endl;
      }
*/
      /*
	TObject* obj = gPad->GetPrimitive("overLego");
      if ( ! obj ) return;
      if ( TH2F* h = dynamic_cast<TH2F*>(obj) )	{
	 h->SetBinContent(
      */
			   /*
   }
   
}

void FWDisplayEvent::pixel2wc(const Int_t pixelX, const Int_t pixelY, 
			      Double_t& wcX, Double_t& wcY, const Double_t wcZ)
{
   // we need to make Pixel to NDC to WC transformation with the following constraint:
   // - Pixel only gives 2 coordinates, so we don't know z coordinate in NDC
   // - We know that in WC z has specific value (depending on what we want to use as 
   //   a selection point. In the case of the base of each bin, z(wc) = 0
   // we need to solve some simple linear equations to get what we need
   Double_t ndcX, ndcY;
   ((TPad *)gPad)->AbsPixeltoXY( pixelX, pixelY, ndcX, ndcY); // Pixel to NDC
   Double_t* m = gPad->GetView()->GetTback(); // NDC to WC matrix
   double part1 = wcZ-m[11]-m[8]*ndcX-m[9]*ndcY;
   wcX = m[3] + m[0]*ndcX + m[1]*ndcY + m[2]/m[10]*part1;
   wcY = m[7] + m[4]*ndcX + m[5]*ndcY + m[6]/m[10]*part1;
}
*/
