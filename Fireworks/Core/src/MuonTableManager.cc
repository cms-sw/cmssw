#include "TableManagers.h"

std::string MuonTableManager::titles[] = {
     "pt"	,
     "global"	,
     "tk"	,
     "SA"	,
     "calo"	,
     "iso(3)"	,
     "iso(5)"	,
     "tr pt"	,
     "eta"	,
     " phi"	,
     "chi^2/ndof"	,
     "matches"	,
     "d0"	,
     "sig(d0)"	,
     " loose(match)"	,
     "tight(match)"	,
     "loose(depth)"	,
     "tight(depth)"
};

std::string MuonTableManager::formats[] = {
     "%5.1f"	,
     "%c"	,
     "%c"	,
     "%c"	,
     "%c"	,
     "%6.3f"	,
     "%6.3f"	,
     "%6.3f"	,
     "%6.3f"	,
     "%6.3f"	,
     "%6.3f"	,
     "%d"	,
     "%6.3f"	,
     "%7.3f"	,
     "%c"	,
     "%c"	,
     "%c"	,
     "%c"
};

int MuonTableManager::NumberOfRows() const
{
     return rows.size();
}

int MuonTableManager::NumberOfCols() const
{
     return sizeof(titles) / sizeof(std::string);
}

struct sort_asc {
     int i;
     bool order;
     bool operator () (const MuonRow &r1, const MuonRow &r2) const 
	  {
	       if (order)
		    return r1.vec()[i] > r2.vec()[i];
	       else return r1.vec()[i] < r2.vec()[i];
	  }
};

void MuonTableManager::Sort(int col, bool sortOrder)
{
     sort_asc sort_fun;
     sort_fun.i = col;
     sort_fun.order = sortOrder;
     std::sort(rows.begin(), rows.end(), sort_fun);
}

std::vector<std::string> MuonTableManager::GetTitles(int col)
{
     std::vector<std::string> ret;
     ret.insert(ret.begin(), titles + col, titles + NumberOfCols());
     return ret;
}

void MuonTableManager::FillCells(int rowStart, int colStart, 
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

TGFrame* MuonTableManager::GetRowCell(int row, TGFrame *parentFrame)
{
     TGTextEntry *cell = new TGTextEntry(format_string("%d", row) ,parentFrame);
     return cell;
}

void MuonTableManager::UpdateRowCell(int row, TGFrame *rowCell)
{
    rowCell->Clear();
    TGTextEntry *cell = (TGTextEntry *)(rowCell);
    cell->SetText(format_string("%d", row).c_str());
}

const std::vector<std::string> 	&MuonRow::str () const
{
     if (str_.size() == 0) {
	  // cache
	  int i = 0;
	  str_.push_back(format_string(MuonTableManager::formats[i++], pt         	));
	  str_.push_back(format_string(MuonTableManager::formats[i++], global     	));
	  str_.push_back(format_string(MuonTableManager::formats[i++], tk         	));
	  str_.push_back(format_string(MuonTableManager::formats[i++], SA         	));
	  str_.push_back(format_string(MuonTableManager::formats[i++], calo       	));
	  str_.push_back(format_string(MuonTableManager::formats[i++], iso_3      	));
	  str_.push_back(format_string(MuonTableManager::formats[i++], iso_5      	));
	  str_.push_back(format_string(MuonTableManager::formats[i++], tr_pt      	));
	  str_.push_back(format_string(MuonTableManager::formats[i++], eta        	));
	  str_.push_back(format_string(MuonTableManager::formats[i++], phi        	));
	  str_.push_back(format_string(MuonTableManager::formats[i++], chi2_ndof  	));
	  str_.push_back(format_string(MuonTableManager::formats[i++], matches    	));
	  str_.push_back(format_string(MuonTableManager::formats[i++], d0         	));
	  str_.push_back(format_string(MuonTableManager::formats[i++], sig_d0     	));
	  str_.push_back(format_string(MuonTableManager::formats[i++], loose_match	));
	  str_.push_back(format_string(MuonTableManager::formats[i++], tight_match	));
	  str_.push_back(format_string(MuonTableManager::formats[i++], loose_depth	));
	  str_.push_back(format_string(MuonTableManager::formats[i++], tight_depth	));
     }
     return str_;
}

const std::vector<float> 	&MuonRow::vec () const
{
     if (vec_.size() == 0) {
	  // cache
	  vec_.push_back(pt         	);
	  vec_.push_back(global     	);
	  vec_.push_back(tk         	);
	  vec_.push_back(SA         	);
	  vec_.push_back(calo       	);
	  vec_.push_back(iso_3      	);
	  vec_.push_back(iso_5      	);
	  vec_.push_back(tr_pt      	);
	  vec_.push_back(eta        	);
	  vec_.push_back(phi        	);
	  vec_.push_back(chi2_ndof  	);
	  vec_.push_back(matches    	);
	  vec_.push_back(d0         	);
	  vec_.push_back(sig_d0     	);
	  vec_.push_back(loose_match	);
	  vec_.push_back(tight_match	);
	  vec_.push_back(loose_depth	);
	  vec_.push_back(tight_depth	);
     }
     return vec_;
}
