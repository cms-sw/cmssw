#include "CommonTools/TrackerMap/interface/VisTrackerMap.h"
#include "CommonTools/TrackerMap/interface/VisTrackerMapBox.h"
#include <qtimer.h>
#include <qpainter.h>
#include <qbitmap.h>
#include <iostream>
using namespace std;

VisTrackerMap::VisTrackerMap( QWidget *parent, const char *name, QLabel* labelinfo )
  : QWidget( parent, name )
{
  setMinimumSize( 3000, 1360 );
  setMaximumSize( 3000, 1360 );
  
  setPalette( QPalette( QColor( 250, 250, 250) ) );
  xmin = -3.0; xmax = 3.0; ymin = -0.1; ymax =1.0;
  ql = labelinfo;

  horizontal_view = true;
  posrel= true;
  tk =new TrackerMap(name,340,200);
  int number_mod = tk->getNumMod();
  reg_mod = new QPointArray[number_mod];
  int count=0;
  
  for (int layer=1; layer < 44; layer++){
    int n_lay=layer;
    tk->defwindow(n_lay);
    for (int ring=tk->firstRing[layer-1]; ring < tk->ntotRing[layer-1]+tk->firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = tk->smoduleMap[key];
        if(mod !=0 && !mod->notInUse()){
	  reg_mod[count] =  QPointArray(4);
	  mod->setQPointArray(count);
	  visDrawModule(mod,mod->getKey(),mod->layer,reg_mod[count]);
	  count++;
        }
      }
    }
  }
}

VisTrackerMap::~VisTrackerMap ()
{

}


void VisTrackerMap::setPaintOptions(bool vtk_print_total,float vtk_minval, float vtk_maxval)
{
  print_total =vtk_print_total;
  minval = vtk_minval;
  maxval = vtk_maxval;
}

void VisTrackerMap::paintEvent( QPaintEvent * )
{
  QPainter *p = new QPainter( this );
  
  if(!print_total){
    for (int layer=1; layer < 44; layer++){
      for (int ring=tk->firstRing[layer-1]; ring < tk->ntotRing[layer-1]+tk->firstRing[layer-1];ring++){
        for (int module=1;module<200;module++) {
          int key=layer*100000+ring*1000+module;
          TmModule * mod = tk->smoduleMap[key];
          if(mod !=0 && !mod->notInUse()){
            mod->value = mod->value / mod->count;
          }
        }
      }
    }
  }
  if(minval>=maxval){
    minval=9999999.;
    maxval=-9999999.;
    for (int layer=1; layer < 44; layer++){
      for (int ring=tk->firstRing[layer-1]; ring < tk->ntotRing[layer-1]+tk->firstRing[layer-1];ring++){
        for (int module=1;module<200;module++) {
          int key=layer*100000+ring*1000+module;
          TmModule * mod = tk->smoduleMap[key];
          if(mod !=0 && !mod->notInUse()){
            if (minval > mod->value)minval=mod->value;
            if (maxval < mod->value)maxval=mod->value;
          }
        }
      }
    }
  }
  for (int layer=1; layer < 44; layer++){
    int n_lay=layer;
    tk->defwindow(n_lay);
    for (int ring=tk->firstRing[layer-1]; ring < tk->ntotRing[layer-1]+tk->firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = tk->smoduleMap[key];
        if(mod !=0 && !mod->notInUse()){
	  computeColor(mod, print_total, p);
	  p->drawPolygon(reg_mod[mod->getQPointArray()]);
        }
      }
    }
  }
}

void VisTrackerMap::mousePressEvent(QMouseEvent *e)
{
  int layer = tk->find_layer(e->pos().x(),e->pos().y());
  cout << "mouse position " << e->pos().x() << " " << e->pos().y()<<" layer "<< layer << endl;
  ostringstream outs;
     
  TmModule * mod;
  for (int ring=tk->firstRing[layer-1]; ring < tk->ntotRing[layer-1]+tk->firstRing[layer-1];ring++){
    for (int module=1;module<200;module++) {
      int key=layer*100000+ring*1000+module;
      mod = tk->smoduleMap[key];
      if(mod !=0 && !mod->notInUse()){

	QRegion q(reg_mod[mod->getQPointArray()],false);
//	QPoint pt(e->pos().x(), (tk->getxsize()*4)-e->pos().y());
	QPoint pt(e->pos().x(), e->pos().y());
	if(q.contains(pt)){

	  outs << mod->name<<" "<< mod->text<<" DetId="<<mod->idex<<" count="<<mod->count<<" value=" <<mod->value<< "mouse pos= "<< e->pos().x() <<" "<<e->pos().y();
	  ql->setText(outs.str());

	  emit moduleSelected(mod->idex);
	  break;
	}
      }
    }
  }
}

void VisTrackerMap::computeColor(TmModule * mod, bool print_total, QPainter* p)
{
  int green = 0;
  
  if(mod->red < 0){ //use count to compute color
    green = (int)((mod->value-minval)/(maxval-minval)*256.);
    
    if (green > 255) green=255;
    if(!print_total)mod->value=mod->value*mod->count;//restore mod->value

    if(mod->count > 0){
      p->setBrush(QColor(255,255-green,0));
      p->setPen(QColor(255,255-green,0));
    }
    else{
      p->setBrush(QColor(white));
    }

  } else {//color defined with fillc
    if(mod->red>255)mod->red=255;
    if(mod->green>255)mod->green=255;
    if(mod->blue>255)mod->blue=255;
    
    p->setBrush(QColor(mod->red, mod->green,mod->blue ));
  }
}

void VisTrackerMap::visDrawModule(TmModule * mod, int key,int nlay,QPointArray a )
{
  int x,y;
  double phi,r,dx,dy, dy1;
  double xp[4],yp[4],xp1,yp1;
  double vhbot,vhtop,vhapo;
  double rmedio[]={0.041,0.0701,0.0988,0.255,0.340,0.430,0.520,0.610,0.696,0.782,0.868,0.965,1.080};
  double xt1,yt1,xs1=0.,ys1=0.,xt2,yt2,xs2,ys2,pv1,pv2;
  int np = 4;
  int numod=0;
  
  phi = tk->phival(mod->posx,mod->posy);
  r = sqrt(mod->posx*mod->posx+mod->posy*mod->posy);
  vhbot = mod->width;
  vhtop=mod->width;
  vhapo=mod->length;
  if(nlay < 31){ //endcap
    vhbot = mod->widthAtHalfLength/2.-(mod->width/2.-mod->widthAtHalfLength/2.);
    vhtop=mod->width/2.;
    vhapo=mod->length/2.;
    if(nlay >12 && nlay <19){
      if(posrel)r = r+r;
      xp[0]=r-vhtop;yp[0]=-vhapo;
      xp[1]=r+vhtop;yp[1]=-vhapo;
      xp[2]=r+vhtop;yp[2]=vhapo;
      xp[3]=r-vhtop;yp[3]=vhapo;
    }else{
      if(posrel)r = r + r/3.;
      xp[0]=r-vhapo;yp[0]=-vhbot;
      xp[1]=r+vhapo;yp[1]=-vhtop;
      xp[2]=r+vhapo;yp[2]=vhtop;
      xp[3]=r-vhapo;yp[3]=vhbot;
    }
    for(int j=0;j<4;j++){
      xp1 = xp[j]*cos(phi)-yp[j]*sin(phi);
      yp1 = xp[j]*sin(phi)+yp[j]*cos(phi);
      xp[j] = xp1;yp[j]=yp1;
    }
  } else { //barrel
    numod=mod->idModule;if(numod>100)numod=numod-100;
    int vane = mod->ring;
    if(posrel){
      dx = vhapo;
      phi=M_PI;
      xt1=rmedio[nlay-31]; yt1=-vhtop/2.;
      xs1 = xt1*cos(phi)-yt1*sin(phi);
      ys1 = xt1*sin(phi)+yt1*cos(phi);
      xt2=rmedio[nlay-31]; yt2=vhtop/2.;
      xs2 = xt2*cos(phi)-yt2*sin(phi);
      ys2 = xt2*sin(phi)+yt2*cos(phi);
      dy=tk->phival(xs2,ys2)-tk->phival(xs1,ys1);
         dy1 = dy;
      if(nlay==31)dy1=0.39;
      if(nlay==32)dy1=0.23;
      if(nlay==33)dy1=0.16;
      xp[0]=vane*(dx+dx/8.);yp[0]=numod*(dy1);
      xp[1]=vane*(dx+dx/8.)+dx;yp[1]=numod*(dy1);
      xp[2]=vane*(dx+dx/8.)+dx;yp[2]=numod*(dy1)+dy;
      xp[3]=vane*(dx+dx/8.);yp[3]=numod*(dy1)+dy;
    }else{
      xt1=r; yt1=-vhtop/2.;
      xs1 = xt1*cos(phi)-yt1*sin(phi);
      ys1 = xt1*sin(phi)+yt1*cos(phi);
      xt2=r; yt2=vhtop/2.;
      xs2 = xt2*cos(phi)-yt2*sin(phi);
      ys2 = xt2*sin(phi)+yt2*cos(phi);
      pv1=tk->phival(xs1,ys1);
      pv2=tk->phival(xs2,ys2);
      if(fabs(pv1-pv2)>M_PI && numod==1)pv1=pv1-2.*M_PI;
      if(fabs(pv1-pv2)>M_PI && numod!=1)pv2=pv2+2.*M_PI;
      xp[0]=mod->posz-vhapo/2.;yp[0]=4.2*pv1;
      xp[1]=mod->posz+vhapo/2.;yp[1]=4.2*pv1;
      xp[2]=mod->posz+vhapo/2. ;yp[2]=4.2*pv2;
      xp[3]=mod->posz-vhapo/2.;yp[3]=4.2*pv2;
    }
  }
  if(tk->isRingStereo(key))
    {
      np = 3;
      if(mod->idModule>100 ){for(int j=0;j<3;j++){
        x=(int)(tk->xdpixel(xp[j]));y=(int)(tk->ydpixel(yp[j]));
        if(!horizontal_view)a.setPoint(j,x,y);else a.setPoint(j,y,1360-x);
      }
      if(!horizontal_view)a.setPoint(3,x,y);else a.setPoint(3,y,1360-x);
      }else {
        x=(int)(tk->xdpixel(xp[2]));y=(int)(tk->ydpixel(yp[2]));
        if(!horizontal_view)a.setPoint(0,x,y); else a.setPoint(0,y,1360-x);
        x=(int)(tk->xdpixel(xp[3]));y=(int)(tk->ydpixel(yp[3]));
        if(!horizontal_view)a.setPoint(1,x,y); else a.setPoint(1,y,1360-x);
        x=(int)(tk->xdpixel(xp[0]));y=(int)(tk->ydpixel(yp[0]));
        if(!horizontal_view)a.setPoint(2,x,y); else a.setPoint(2,y,1360-x);
        if(!horizontal_view)a.setPoint(3,x,y); else a.setPoint(3,y,1360-x);
      }
    } else {
      for(int j=0;j<4;j++){
        x=(int)(tk->xdpixel(xp[j]));y=(int)(tk->ydpixel(yp[j]));
        if(!horizontal_view)a.setPoint(j,x,y);else a.setPoint(j,y,1360-x);
      }
    }
}
