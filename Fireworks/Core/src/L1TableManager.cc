#include "TableManagers.h"

std::string L1TableManager::titles[] = {
     "Et"	,
     "eta"	,
     "phi"	,
};

std::string L1TableManager::formats[] = {
     "%5.1f"	,
     "%6.3f"	,
     "%6.3f"	,
};

int L1TableManager::NumberOfRows() const
{
     return rows.size();
}

int L1TableManager::NumberOfCols() const
{
     return sizeof(titles) / sizeof(std::string);
}

void L1TableManager::Sort(int col, bool sortOrder)
{
     sort_asc<L1Row> sort_fun(this);
     sort_fun.i = col;
     sort_fun.order = sortOrder;
     std::sort(rows.begin(), rows.end(), sort_fun);
}

std::vector<std::string> L1TableManager::GetTitles(int col)
{
     std::vector<std::string> ret;
     ret.insert(ret.begin(), titles + col, titles + NumberOfCols());
     return ret;
}

void L1TableManager::FillCells(int rowStart, int colStart,
				     int rowEnd, int colEnd,
				     std::vector<std::string> &ret)
{
     ret.clear();
     ret.reserve((rowEnd - rowStart) * (colEnd - colStart));
     for (int i = rowStart; i < rowEnd && i < NumberOfRows(); ++i) {
	  const std::vector<std::string> &row = rows[i].str();
	  if ((unsigned int)colEnd > row.size()) {
	       ret.insert(ret.end(),
			  row.begin() + colStart, row.end());
	       ret.insert(ret.end(), colEnd - row.size(), "");
	  } else {
	       ret.insert(ret.end(),
			  row.begin() + colStart, row.begin() + colEnd);
	  }
     }
     // no, don't return ret;
}

TGFrame* L1TableManager::GetRowCell(int row, TGFrame *parentFrame)
{
     TGTextEntry *cell = new TGTextEntry(format_string("%d", row) ,parentFrame);
     return cell;
}

void L1TableManager::UpdateRowCell(int row, TGFrame *rowCell)
{
    rowCell->Clear();
    TGTextEntry *cell = (TGTextEntry *)(rowCell);
    cell->SetText(format_string("%d", row).c_str());
}

const std::vector<std::string> 	&L1Row::str () const
{
     if (str_.size() == 0) {
	  // cache
	  int i = 0;
	  str_.push_back(format_string(L1TableManager::formats[i++], Et                 ));
	  str_.push_back(format_string(L1TableManager::formats[i++], eta                ));
	  str_.push_back(format_string(L1TableManager::formats[i++], phi                ));
     }
     return str_;
}

const std::vector<float> 	&L1Row::vec () const
{
     if (vec_.size() == 0) {
	  // cache
	  vec_.push_back(Et                 );
	  vec_.push_back(eta                );
	  vec_.push_back(phi                );
     }
     return vec_;
}
