#ifndef VIS_TRACKERMAP_H
#define VIS_TRACKERMAP_H

#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CommonTools/TrackerMap/interface/TmModule.h"

#include <qwidget.h>
#include <qdatetime.h>
#include <Qt3Support/q3pointarray.h>
#include <qtooltip.h>
#include <qlabel.h>
#include <map>


class VisTrackerMap : public QWidget		
{
  Q_OBJECT
      public:
    VisTrackerMap( QWidget *parent=0, const char *name=0 , QLabel *labelinfo=0);
    ~VisTrackerMap();
    
    void paintEvent(QPaintEvent* );
    void mousePressEvent(QMouseEvent *e);
    void visDrawModule(TmModule * mod, int key,int nlay, Q3PointArray a);
    void computeColor(TmModule * mod, bool print_total, QPainter* p);
    void setPaintOptions(bool vtk_print_total=true,float vtk_minval=0., float vtk_maxval=0.);
    
    TrackerMap* getTrackerMap(){return tk;}
    TrackerMap *tk;
    QLabel* ql;

 signals:
    void moduleSelected(int);

 private:
    
    float minval,maxval;
    bool print_total;
    bool posrel,horizontal_view;
    double xmin,xmax,ymin,ymax;
 
    Q3PointArray *reg_mod;
};

#endif 
