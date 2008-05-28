#define private public
#include "TableWidget.h"
#undef private
#include <string.h>
#include "TableManagers.h"

std::string format_string (const std::string &fmt, int x)
{
     if (fmt == "%c") {
	  switch (x) {
	  case FLAG_YES:
	       return "yes";
	  case FLAG_NO:
	       return "no";
	  case FLAG_MAYBE:
	       return "maybe";
	  default:
	       return "unknown";
	  }
     } else {
	  char str[100];
	  snprintf(str, 100, fmt.c_str(), x);
	  return str;
     }
}

std::string format_string (const std::string &fmt, double x)
{
     char str[100];
     snprintf(str, 100, fmt.c_str(), x);
     return str;
}

void FWTableManager::MakeFrame (TGMainFrame *parent, int width, int height) 
{
     // display the table name prominently
     TGTextEntry *m_tNameEntry = new TGTextEntry(title().c_str(), parent);
     TGLayoutHints *m_tNameHints = new TGLayoutHints(
	  kLHintsCenterX |
	  kLHintsExpandX |
	  kLHintsFillX );
     parent->AddFrame(m_tNameEntry, m_tNameHints);

     frame = new TGCompositeFrame(parent, width, height);
     TGLayoutHints *tFrameHints = 
 	  new TGLayoutHints(kLHintsTop|kLHintsLeft|
 			    kLHintsExpandX);
     parent->AddFrame(frame,tFrameHints);
     parent->HideFrame(frame);
     
     widget = new TableWidget(frame, this); 
     m_tNameEntry->Resize(width, widget->m_cellHeight);
     m_tNameEntry->SetBackgroundColor(widget->m_titleColor);
     m_tNameEntry->SetAlignment(kTextCenterX);
     m_tNameEntry->ChangeOptions(kRaisedFrame);
     title_frame = m_tNameEntry;
//      widget->HighlightRow(0);
}
