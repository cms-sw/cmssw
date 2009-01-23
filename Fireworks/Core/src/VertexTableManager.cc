#include "TableManagers.h"

std::string VertexTableManager::titles[] = {
   "index",
   "vx",
   "vx err",
   "vy",
   "vy err",
   "vz",
   "vz err",
   "tracks",
   "chi2",
   "ndof",
};

std::string VertexTableManager::formats[] = {
   "%d",
   "%6.3f",
   "%6.3f",
   "%6.3f",
   "%6.3f",
   "%6.3f",
   "%6.3f",
   "%d",
   "%6.3f",
   "%6.3f",
};

int VertexTableManager::NumberOfRows() const
{
   return rows.size();
}

int VertexTableManager::NumberOfCols() const
{
   return sizeof(titles) / sizeof(std::string);
}

void VertexTableManager::Sort(int col, bool sortOrder)
{
   sort_asc<VertexRow> sort_fun(this);
   sort_fun.i = col;
   sort_fun.order = sortOrder;
   std::sort(rows.begin(), rows.end(), sort_fun);
}

std::vector<std::string> VertexTableManager::GetTitles(int col)
{
   std::vector<std::string> ret;
   ret.insert(ret.begin(), titles + col, titles + NumberOfCols());
   return ret;
}

void VertexTableManager::FillCells(int rowStart, int colStart,
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

TGFrame* VertexTableManager::GetRowCell(int row, TGFrame *parentFrame)
{
   TGTextEntry *cell = new TGTextEntry(format_string("%d", row),parentFrame);
   return cell;
}

void VertexTableManager::UpdateRowCell(int row, TGFrame *rowCell)
{
   rowCell->Clear();
   TGTextEntry *cell = (TGTextEntry *)(rowCell);
   cell->SetText(format_string("%d", row).c_str());
}

const std::vector<std::string>  &VertexRow::str () const
{
   if (str_.size() == 0) {
      // cache
      int i = 0;
      str_.push_back(format_string(VertexTableManager::formats[i++], index          ));
      str_.push_back(format_string(VertexTableManager::formats[i++], vx             ));
      str_.push_back(format_string(VertexTableManager::formats[i++], vx_err         ));
      str_.push_back(format_string(VertexTableManager::formats[i++], vy             ));
      str_.push_back(format_string(VertexTableManager::formats[i++], vy_err         ));
      str_.push_back(format_string(VertexTableManager::formats[i++], vz             ));
      str_.push_back(format_string(VertexTableManager::formats[i++], vz_err         ));
      str_.push_back(format_string(VertexTableManager::formats[i++], n_tracks       ));
      str_.push_back(format_string(VertexTableManager::formats[i++], chi2           ));
      str_.push_back(format_string(VertexTableManager::formats[i++], ndof           ));
   }
   return str_;
}

const std::vector<float>        &VertexRow::vec () const
{
   if (vec_.size() == 0) {
      // cache
      vec_.push_back(index          );
      vec_.push_back(vx             );
      vec_.push_back(vx_err         );
      vec_.push_back(vy             );
      vec_.push_back(vy_err         );
      vec_.push_back(vz             );
      vec_.push_back(vz_err         );
      vec_.push_back(n_tracks       );
      vec_.push_back(chi2           );
      vec_.push_back(ndof           );
   }
   return vec_;
}
