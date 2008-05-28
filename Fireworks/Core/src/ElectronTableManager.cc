#include "TableManagers.h"

std::string ElectronTableManager::titles[] = {
     "Et"	,
     "eta"	,
     "phi"	,
     "E/p"	,
     "H/E"	,
     "fbrem"	,
     "dei"	,
     "dpi"	,
     "see"	,
     "spp"	,
     "iso"	,
     "robust"	,
     "loose"	,
     "tight"
};

std::string ElectronTableManager::formats[] = {
     "%5.1f"	,
     "%6.3f"	,
     "%6.3f"	,
     "%6.3f"	,
     "%6.3f"	,
     "%6.3f"	,
     "%6.3f"	,
     "%6.3f"	,
     "%6.3f"	,
     "%6.3f"	,
     "%6.3f"	,
     "%c"	,
     "%c"	,
     "%c"
};

int ElectronTableManager::NumberOfRows() const
{
     return rows.size();
}

int ElectronTableManager::NumberOfCols() const
{
     return sizeof(titles) / sizeof(std::string);
}

struct sort_asc {
     int i;
     bool order;
     bool operator () (const ElectronRow &r1, const ElectronRow &r2) const 
	  {
	       if (order)
		    return r1.vec()[i] > r2.vec()[i];
	       else return r1.vec()[i] < r2.vec()[i];
	  }
};

void ElectronTableManager::Sort(int col, bool sortOrder)
{
     sort_asc sort_fun;
     sort_fun.i = col;
     sort_fun.order = sortOrder;
     std::sort(rows.begin(), rows.end(), sort_fun);
}

std::vector<std::string> ElectronTableManager::GetTitles(int col)
{
     std::vector<std::string> ret;
     ret.insert(ret.begin(), titles + col, titles + NumberOfCols());
     return ret;
}

void ElectronTableManager::FillCells(int rowStart, int colStart, 
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

TGFrame* ElectronTableManager::GetRowCell(int row, TGFrame *parentFrame)
{
     TGTextEntry *cell = new TGTextEntry(format_string("%d", row) ,parentFrame);
     return cell;
}

void ElectronTableManager::UpdateRowCell(int row, TGFrame *rowCell)
{
    rowCell->Clear();
    TGTextEntry *cell = (TGTextEntry *)(rowCell);
    cell->SetText(format_string("%d", row).c_str());
}

const std::vector<std::string> 	&ElectronRow::str () const
{
     if (str_.size() == 0) {
	  // cache
	  int i = 0;
	  str_.push_back(format_string(ElectronTableManager::formats[i++], Et    	));
	  str_.push_back(format_string(ElectronTableManager::formats[i++], eta   	));
	  str_.push_back(format_string(ElectronTableManager::formats[i++], phi   	));
	  str_.push_back(format_string(ElectronTableManager::formats[i++], eop   	));
	  str_.push_back(format_string(ElectronTableManager::formats[i++], hoe   	));
	  str_.push_back(format_string(ElectronTableManager::formats[i++], fbrem 	));
	  str_.push_back(format_string(ElectronTableManager::formats[i++], dei   	));
	  str_.push_back(format_string(ElectronTableManager::formats[i++], dpi   	));
	  str_.push_back(format_string(ElectronTableManager::formats[i++], see   	));
	  str_.push_back(format_string(ElectronTableManager::formats[i++], spp   	));
	  str_.push_back(format_string(ElectronTableManager::formats[i++], iso  	));
	  str_.push_back(format_string(ElectronTableManager::formats[i++], robust	));
	  str_.push_back(format_string(ElectronTableManager::formats[i++], loose 	));
	  str_.push_back(format_string(ElectronTableManager::formats[i++], tight	));
     }
     return str_;
}

const std::vector<float> 	&ElectronRow::vec () const
{
     if (vec_.size() == 0) {
	  // cache
	  vec_.push_back(Et    	);
	  vec_.push_back(eta   	);
	  vec_.push_back(phi   	);
	  vec_.push_back(eop   	);
	  vec_.push_back(hoe   	);
	  vec_.push_back(fbrem 	);
	  vec_.push_back(dei   	);
	  vec_.push_back(dpi   	);
	  vec_.push_back(see   	);
	  vec_.push_back(spp   	);
	  vec_.push_back(iso  	);
	  vec_.push_back(robust	);
	  vec_.push_back(loose 	);
	  vec_.push_back(tight	);
     }
     return vec_;
}

