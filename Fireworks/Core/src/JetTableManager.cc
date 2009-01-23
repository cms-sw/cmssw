#include "TableManagers.h"

std::string JetTableManager::titles[] = {
   "Et",
   " eta",
   " phi",
   " ECAL",
   " HCAL",
   " emf",
//      " chf"
};

std::string JetTableManager::formats[] = {
   "%5.1f",
   " %6.3f",
   " %6.3f",
   " %5.1f",
   " %5.1f",
   " %6.3f",
//      " %6.3f"
};

int JetTableManager::NumberOfRows() const
{
   return rows.size();
}

int JetTableManager::NumberOfCols() const
{
   return sizeof(titles) / sizeof(std::string);
}

void JetTableManager::Sort(int col, bool sortOrder)
{
   sort_asc<JetRow> sort_fun(this);
   sort_fun.i = col;
   sort_fun.order = sortOrder;
   std::sort(rows.begin(), rows.end(), sort_fun);
}

std::vector<std::string> JetTableManager::GetTitles(int col)
{
   std::vector<std::string> ret;
   ret.insert(ret.begin(), titles + col, titles + NumberOfCols());
   return ret;
}

void JetTableManager::FillCells(int rowStart, int colStart,
                                int rowEnd, int colEnd,
                                std::vector<std::string> &ret)
{
   ret.clear();
   ret.reserve((rowEnd - rowStart) * (colEnd - colStart));
   for (int i = rowStart; i < rowEnd; ++i) {
      if (i < NumberOfRows()) {
         const std::vector<std::string> &row = rows[i].str();
         if ((unsigned int)colEnd > row.size()) {
            ret.insert(ret.end(),
                       row.begin() + colStart, row.end());
            ret.insert(ret.end(), colEnd - row.size(), "");
         } else {
            ret.insert(ret.end(),
                       row.begin() + colStart, row.begin() + colEnd);
         }
      } else {
         for (int j = colStart; j < colEnd; ++j) {
            ret.push_back("");
         }
      }
   }
   // no, don't return ret;
}

TGFrame* JetTableManager::GetRowCell(int row, TGFrame *parentFrame)
{
   TGTextEntry *cell = new TGTextEntry(format_string("%d", row),parentFrame);
   return cell;
}

void JetTableManager::UpdateRowCell(int row, TGFrame *rowCell)
{
   rowCell->Clear();
   TGTextEntry *cell = (TGTextEntry *)(rowCell);
   cell->SetText(format_string("%d", row).c_str());
}

const std::vector<std::string>  &JetRow::str () const
{
   if (str_.size() == 0) {
      // cache
      int i = 0;
      str_.push_back(format_string(JetTableManager::formats[i++], Et        ));
      str_.push_back(format_string(JetTableManager::formats[i++], eta       ));
      str_.push_back(format_string(JetTableManager::formats[i++], phi       ));
      str_.push_back(format_string(JetTableManager::formats[i++], ECAL      ));
      str_.push_back(format_string(JetTableManager::formats[i++], HCAL      ));
      str_.push_back(format_string(JetTableManager::formats[i++], emf       ));
//        str_.push_back(format_string(JetTableManager::formats[i++], chf	));
   }
   return str_;
}

const std::vector<float>        &JetRow::vec () const
{
   if (vec_.size() == 0) {
      // cache
      vec_.push_back(Et     );
      vec_.push_back(eta    );
      vec_.push_back(phi    );
      vec_.push_back(ECAL   );
      vec_.push_back(HCAL   );
      vec_.push_back(emf    );
//        vec_.push_back(chf	);
   }
   return vec_;
}
