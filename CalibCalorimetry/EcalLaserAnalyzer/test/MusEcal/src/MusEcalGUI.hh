#ifndef MusEcalGUI_hh
#define MusEcalGUI_hh

//
// MusEcal : Monitoring and Useful Survey of CMS Ecal
//                  Clever Analysis of Laser
//
// Authors  : Gautier Hamel de Monchenault and Julie Malcles, Saclay 
//           Inspired from MONECAL, by F.X. Gentit and P. Verrecchia 
//

#include <TApplication.h>
#include <TVirtualX.h>
#include <TGButton.h>
#include <TGResourcePool.h>
#include <TGListBox.h>
#include <TGListTree.h>
#include <TGFSContainer.h>
#include <TGClient.h>
#include <TGFrame.h>
#include <TGIcon.h>
#include <TGLabel.h>
#include <TGButton.h>
#include <TGTextEntry.h>
#include <TGNumberEntry.h>
#include <TGMsgBox.h>
#include <TGMenu.h>
#include <TGCanvas.h>
#include <TGComboBox.h>
#include <TGTab.h>
#include <TGSlider.h>
#include <TGDoubleSlider.h>
#include <TGFileDialog.h>
#include <TGTextEdit.h>
#include <TGShutter.h>
#include <TGProgressBar.h>
#include <TGColorSelect.h>
#include <TRootEmbeddedCanvas.h>
#include <TRandom.h>
#include <TSystem.h>
#include <TSystemDirectory.h>
#include <TEnv.h>
#include <TFile.h>
#include <TKey.h>
#include <TGDockableFrame.h>
#include <TGFontDialog.h>

#include <map>

// base classes
#include "MEClickableCanvas.hh"
#include "MECanvasHolder.hh"
#include "MusEcal.hh"
class MEPlotWindow;
class MERunPanel;
class MEChanPanel;
class MELeafPanel;
class MEMultiVarPanel;

class MusEcalGUI : public TGMainFrame, public MECanvasHolder, public MusEcal
{
  enum { iHIST, iVS_CHANNEL, iMAP };

public:

  // contructors/destructor
  MusEcalGUI() {}
  MusEcalGUI( const TGWindow *p, UInt_t w, UInt_t h,
	      int type=ME::iLaser, int color=ME::iBlue );
  virtual  ~MusEcalGUI();

protected:

  // exit gracefuly
  void exitProg();

private:

  // virtual MusEcal functions
  virtual void refresh();
  virtual void setType( int type, int color=0 );
  virtual void setLMRegion( int lmr );
  virtual void setTime( ME::Time );
  //  virtual void setChannel( int ig, int ieta, int iphi, bool useEtaPhi=true );
  virtual void setChannel( MEChannel* );

  void welcome();

  bool getTimeVector( std::vector< ME::Time >& );
  bool getHistoryVector( std::vector< ME::Time >&,
			 std::vector< float >&,
			 std::vector< bool >&,
			 std::vector< float >&,
			 bool& b_, float& miny, float& maxy );
  bool getHistoryVector( unsigned int& nrun, 
			 float* x, float* y, float* ex, float* ey,
			 bool* ok, float& norm, float& min, float& max );
  void drawHistoryGraph( unsigned int n, 
			 float* x, float* y, float* ex, float* ey, bool* ok,
			 int markerStyle, float markerSize, int markerColor,
			 int lineWidth, int lineColor, const char* graphOpt="LPSame" );
  
  // Run selection panel
  MERunPanel* _fRunPanel;
  friend class MERunPanel;
  void createRunPanel();

  // Channel selection panel
  MEChanPanel* _fChanPanel;
  friend class MEChanPanel;
  void createChanPanel( bool ifexists=false );

  // leaf panel
  MELeafPanel* _fLeafPanel;
  friend class MELeafPanel;
  void createLeafPanel();

  // multiVar panel
  MEMultiVarPanel* _fMultiVarPanel;
  friend class MEMultiVarPanel;
  void createMultiVarPanel();

  // draw functions
  int _ihtype;
  int _icateg;
  int _ihist;
  TString _psdir;
  void drawHist(    int opt=0 );
  void drawAPDHist( int opt=0 );
  void drawPNHist(  int opt=0 );
  void drawMTQHist( int opt=0 );

  void drawAPDAnim( int opt=0 );

  void historyPlot(  int opt=0 );
  void leafPlot(     int opt=0 );
  void multiVarPlot( int opt=0 );

  // normalize history?
  bool _normalize;  // FIXME: was in MusEcal
  
  // current type of history plot
  enum { iHistoryVsTime, iHistoryProjection };
  int _historyType;

  friend class MEPlotWindow;
  std::map< TString, MEPlotWindow* > _window;
  MEPlotWindow* getWindow( TString, int opt, int w, int h );

  // action after a double click
  friend class MEClickableCanvas;
  virtual void setPxAndPy( int px, int py ); 
  void windowClicked( MEPlotWindow* canv );

  //
  // layouts and menus
  //
  void setupMainWindow();
  void setLMRMenu();

  TGLayoutHints* fMenuBarLayout; 
  TGLayoutHints* fMenuBarItemLayout; 
  TGLayoutHints* fL1;              //Layout of fEcanvas in fHFrame1
  TGLayoutHints* fL2;              //Layout of fHFrame1 in this
  TGLayoutHints* fL5;              //Layout of fHFrame2 in this
  TGLayoutHints* fLb;              //Layout of fVFrame in fHFrame1
  TGLayoutHints* fL8;              //Layout of fVFrame in fHFrame1

  // docks and bars
  TGDockableFrame* fMenuDock;
  TGMenuBar*       fMenuBar;

  // menus
  TGPopupMenu* f_File_Menu;
  TGPopupMenu* f_Hist_Menu;

  // if type is laser
  TGPopupMenu* f_Laser_Menu;
  TGPopupMenu* f_APD_Menu;
  TGPopupMenu* f_PN_Menu;
  TGPopupMenu* f_MTQ_Menu;
  TGPopupMenu* f_APD_Hist_Menu[ME::iSizeC];
  TGPopupMenu* f_PN_Hist_Menu[ME::iSizeC]; 
  TGPopupMenu* f_MTQ_Hist_Menu[ME::iSizeC];

  // if type is testpulse
  TGPopupMenu* f_TP_Menu;
  TGPopupMenu* f_TPAPD_Gain_Menu;
  TGPopupMenu* f_TPPN_Gain_Menu;
  TGPopupMenu* f_TPAPD_Hist_Menu;
  TGPopupMenu* f_TPPN_Hist_Menu;

  TGPopupMenu* f_Channel_Menu;
  std::vector<TGPopupMenu*> f_tree_menu;
  
  // History menu
  TGPopupMenu* f_History_Menu;
  TGPopupMenu* f_History_L_Menu;
  TGPopupMenu* f_History_TPV_Menu;
  TGPopupMenu* f_History_LV_Menu[ME::iSizeC]; 

  // Frames
  TGHorizontalFrame*   fHFrame1;    // Horizontal frame up
  TGHorizontalFrame*   fHFrame2;    // Horizontal frame down
  TGVerticalFrame*     fVFrame;     // Left frame in fHFrame1
  MEClickableCanvas*   fEcanvas;    // RootEmbeddedCanvas in fHFrame1, at right - clickable !

  TPad* _curPad;

public:

  // slots
  void HandleFileMenu(Int_t);
  void HandleHistMenu(Int_t);
  void HandleHistoryMenu(Int_t);
  void HandleChannelMenu(Int_t);

  ClassDef( MusEcalGUI, 0 ) // MusEcalGUI -- Monitoring utility for survey of Ecal
};

#endif

