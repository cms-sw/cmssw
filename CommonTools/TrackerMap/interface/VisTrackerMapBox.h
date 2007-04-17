#include "CommonTools/TrackerMap/interface/VisTrackerMap.h"
#include <qlabel.h>
#include <qwidget.h>

class QGridLayout;

class VisTrackerMapBox : public QWidget
{
  Q_OBJECT
    public:
  VisTrackerMapBox( QWidget *parent = 0, const char *name = 0 );
  ~VisTrackerMapBox() {}
  
  void update();
  TrackerMap* getTrackerMap(){return trackerMap;}
  void setPaintOptions(bool print_total,float minval=0.,float maxval=0.){v_tk->setPaintOptions(print_total, minval, maxval);}
  QLabel * labelinfo;
  VisTrackerMap *v_tk;	
  int getSelectedModule() {return selectedModule; selectedModule = 0;} 
  
 signals:
  void moduleSelected(int);
  
  public slots:
    void emitModSel(int id);
  
 private:
  TrackerMap *trackerMap;
  int selectedModule;
};
