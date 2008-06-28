#define private public
#include "TableWidget.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
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

FWTableManager::FWTableManager () 
     : TableManager(),
       widget	(0),
       frame	(0),
       title_frame	(0),
       item	(0)
{

}

void FWTableManager::MakeFrame (TGCompositeFrame *parent, int width, int height) 
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
//      widget = new TableWidget(frame, this, width, height / 3, 5, 19); 
     m_tNameEntry->Resize(width, widget->m_cellHeight);
     m_tNameEntry->SetBackgroundColor(widget->m_titleColor);
     m_tNameEntry->SetAlignment(kTextCenterX);
     m_tNameEntry->ChangeOptions(kRaisedFrame);
     title_frame = m_tNameEntry;
//      widget->HighlightRow(0);
}

void FWTableManager::Update (int rows)
{
//      widget->InitTableCells(); 
     widget->Reinit(rows);
}

void FWTableManager::Selection (int row, int mask) 
{
     // This function handles the propagation of the table selection
     // to the framework.  For propagation in the opposite direction,
     // see FWTextView::selectionChanged().
     if (row >= NumberOfRows()) // click on an empty line
	  return;
     int index = table_row_to_index(row);
     switch (mask) { 
     case 4: 
     {
	  // toggle new line
	  for (std::set<int>::const_iterator 
		    i = sel_indices.begin(), end = sel_indices.end();
	       i != end; ++i) {
	       printf("selected index %d\n", *i);
	  }
	  std::set<int>::iterator existing_row = sel_indices.find(index);
	  if (existing_row == sel_indices.end()) {
	       // row is not selected, select it
	       printf("selecting index %d\n", index);
	       item->select(index);
	  } else {
	       // row is selected yet, unselect it
	       printf("unselecting index %d\n", index);
	       item->unselect(index);
	  }
	  break;
     }
     default:
 	  // means only this line is selected
	  item->m_selectionManager->clearSelection();
	  item->select(index);
	  break;
//      case 1:
// 	  // select everything between old and new
// 	  break;
     };
     item->m_selectionManager->finishedAllSelections();
}

void FWTableManager::selectRows ()
{
     // highlight whatever rows the framework told us to
     std::set<int> rows;
     for (std::set<int>::const_iterator i = sel_indices.begin(), 
	       end = sel_indices.end(); i != end; ++i) {
	  rows.insert(index_to_table_row(*i));
     }
     if (widget != 0)
	  widget->SelectRows(rows);
}

void FWTableManager::dump (FILE *f)
{
     std::vector<std::string> titles = GetTitles(0);
     std::vector<int> col_width;
     for (std::vector<std::string>::const_iterator i = titles.begin();
	  i != titles.end(); ++i) {
	  col_width.push_back(i->length());
     }
     std::vector<std::string> row_content;
     for (int row = 0; row < NumberOfRows(); ++row) {
	  FillCells(row, 0, row + 1, NumberOfCols(), row_content);
	  for (std::vector<std::string>::const_iterator i = row_content.begin();
	       i != row_content.end(); ++i) {
	       const int length = i->length();
	       if (col_width[i - row_content.begin()] < length)
		    col_width[i - row_content.begin()] = length;
	  }
     }
     int total_len = 0;
     for (unsigned int i = 0; i < col_width.size(); ++i) 
	  total_len += col_width[i] + 1;
     for (int i = 0; i < total_len; ++i)
	  fprintf(f, "="); 
     fprintf(f, "\n");
     fprintf(f, "%*s", (total_len + title().length()) / 2, title().c_str());
     fprintf(f, "\n");
     for (int i = 0; i < total_len; ++i)
	  fprintf(f, "-"); 
     fprintf(f, "\n");
     for (unsigned int i = 0; i < titles.size(); ++i) {
	  fprintf(f, "%*s", col_width[i] + 1, titles[i].c_str());
     }
     fprintf(f, "\n");
     for (int i = 0; i < total_len; ++i)
	  fprintf(f, "-"); 
     fprintf(f, "\n");
     for (int row = 0; row < NumberOfRows(); ++row) {
	  if (row == 20) {
	       const char no_more[] = "more skipped";
	       fprintf(f, "%*d %s\n", (total_len - sizeof(no_more)) / 2, 
		       NumberOfRows() - 20, no_more);
	       break;
	  }
	  FillCells(row, 0, row + 1, NumberOfCols(), row_content);
	  for (unsigned int i = 0; i < row_content.size(); ++i) {
	       fprintf(f, "%*s", col_width[i] + 1, row_content[i].c_str());
	  }
	  fprintf(f, "\n");
     }
     for (int i = 0; i < total_len; ++i)
	  fprintf(f, "="); 
     fprintf(f, "\n\n");
}
