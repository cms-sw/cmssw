#include "CommonTools/TrackerMap/interface/VisTrackerMapBox.h"
#include <qlayout.h>
#include <qscrollview.h>

VisTrackerMapBox::VisTrackerMapBox( QWidget *parent, const char *name )
  : QWidget( parent, name )
{
  selectedModule = 0;

  setCaption( "TrackerMap" );
  resize( 500, 500 );
  
  QGridLayout *mainGrid = new QGridLayout( this, 2, 1, 2 );
  labelinfo = new QLabel(this);
  labelinfo->setText( trUtf8( " " ) );

  QScrollView* sv = new QScrollView(this);
  sv->setGeometry( QRect( 0, 1, 600, 600 ) );
  v_tk = new VisTrackerMap(sv->viewport(),"Tracker Map",labelinfo);
  sv->addChild(v_tk);

  connect(v_tk, SIGNAL(moduleSelected(int)), this, SLOT(emitModSel(int)));

  mainGrid->addWidget( sv, 0, 0 );
  mainGrid->addWidget( labelinfo, 1, 0 );
}

void VisTrackerMapBox::update()
{
  v_tk->repaint();
}

void VisTrackerMapBox::emitModSel(int id)
{
   selectedModule = id;
   emit moduleSelected(id);
}
