#include "CommonTools/TrackerMap/interface/VisTrackerMapBox.h"
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TObjArray.h"
#include "TCanvas.h"
#include "TString.h"
#include "TPostScript.h"
#include <sstream>
#include <string>
#include <qpainter.h>
#include <qpixmap.h>
#include <qlayout.h>
# include  <iostream>
class TmTestWidget : public QWidget 
{
   Q_OBJECT
 public:
   TmTestWidget (QWidget *parent=0, const char *name=0);
   ~TmTestWidget ();
   void paintEvent(QPaintEvent* );
 signals:
 public slots: 
   void write(int moduleGeomdetId);
 private:
 TFile* f;
};
