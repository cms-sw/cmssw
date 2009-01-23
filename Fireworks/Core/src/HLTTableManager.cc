#include "TableManagers.h"

std::string HLTTableManager::titles[] = {
   "pt",
   "eta",
   "phi",
   "d0",
   "d0 err",
   "z0",
   "z0 err",
   "vtx x",
   "vtx y",
   "vtx z",
   "pix layers",
   "strip layers",
   "outermost layer",
   "chi2",
   "ndof",
};

std::string HLTTableManager::formats[] = {
   "%5.1f",
   "%6.3f",
   "%6.3f",
   "%6.3f",
   "%6.3f",
   "%6.3f",
   "%6.3f",
   "%6.3f",
   "%6.3f",
   "%6.3f",
   "%d",
   "%d",
   "%d",
   "%6.3f",
   "%6.3f",
};

int HLTTableManager::NumberOfRows() const
{
   return rows.size();
}

int HLTTableManager::NumberOfCols() const
{
   return sizeof(titles) / sizeof(std::string);
}

void HLTTableManager::Sort(int col, bool sortOrder)
{
   sort_asc<HLTRow> sort_fun(this);
   sort_fun.i = col;
   sort_fun.order = sortOrder;
   std::sort(rows.begin(), rows.end(), sort_fun);
}

std::vector<std::string> HLTTableManager::GetTitles(int col)
{
   std::vector<std::string> ret;
   ret.insert(ret.begin(), titles + col, titles + NumberOfCols());
   return ret;
}

void HLTTableManager::FillCells(int rowStart, int colStart,
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

TGFrame* HLTTableManager::GetRowCell(int row, TGFrame *parentFrame)
{
   TGTextEntry *cell = new TGTextEntry(format_string("%d", row),parentFrame);
   return cell;
}

void HLTTableManager::UpdateRowCell(int row, TGFrame *rowCell)
{
   rowCell->Clear();
   TGTextEntry *cell = (TGTextEntry *)(rowCell);
   cell->SetText(format_string("%d", row).c_str());
}

const std::vector<std::string>  &HLTRow::str () const
{
   if (str_.size() == 0) {
      // cache
      int i = 0;
      str_.push_back(format_string(HLTTableManager::formats[i++], pt                 ));
      str_.push_back(format_string(HLTTableManager::formats[i++], eta                ));
      str_.push_back(format_string(HLTTableManager::formats[i++], phi                ));
      str_.push_back(format_string(HLTTableManager::formats[i++], d0                 ));
      str_.push_back(format_string(HLTTableManager::formats[i++], d0_err             ));
      str_.push_back(format_string(HLTTableManager::formats[i++], z0                 ));
      str_.push_back(format_string(HLTTableManager::formats[i++], z0_err             ));
      str_.push_back(format_string(HLTTableManager::formats[i++], vtx_x              ));
      str_.push_back(format_string(HLTTableManager::formats[i++], vtx_y              ));
      str_.push_back(format_string(HLTTableManager::formats[i++], vtx_z              ));
      str_.push_back(format_string(HLTTableManager::formats[i++], pix_layers         ));
      str_.push_back(format_string(HLTTableManager::formats[i++], strip_layers       ));
      str_.push_back(format_string(HLTTableManager::formats[i++], outermost_layer    ));
      str_.push_back(format_string(HLTTableManager::formats[i++], chi2               ));
      str_.push_back(format_string(HLTTableManager::formats[i++], ndof               ));
   }
   return str_;
}

const std::vector<float>        &HLTRow::vec () const
{
   if (vec_.size() == 0) {
      // cache
      vec_.push_back(pt                 );
      vec_.push_back(eta                );
      vec_.push_back(phi                );
      vec_.push_back(d0                 );
      vec_.push_back(d0_err             );
      vec_.push_back(z0                 );
      vec_.push_back(z0_err             );
      vec_.push_back(vtx_x              );
      vec_.push_back(vtx_y              );
      vec_.push_back(vtx_z              );
      vec_.push_back(pix_layers         );
      vec_.push_back(strip_layers       );
      vec_.push_back(outermost_layer    );
      vec_.push_back(chi2               );
      vec_.push_back(ndof               );
   }
   return vec_;
}
