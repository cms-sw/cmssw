#include "TH2F.h"
#include <string>
#include <cmath>

inline int countEmptyRows(std::vector<float>& vec) {
  int cnt = 0;
  for (std::vector<float>::const_iterator it = vec.begin(); it != vec.end(); it++)
    if ((*it) == 0.0f)
      cnt++;

  return cnt;
}

inline void fillFunctionParamsValues(
    TH2F*& align, std::vector<float>& m_params, std::string title, int& gridRows, int& NbColumns) {
  const int maxInCol = 25;
  int NbRows = 0;

  NbRows = m_params.size() - countEmptyRows(m_params);

  gridRows = (NbRows <= maxInCol) ? NbRows : maxInCol;
  NbColumns = ceil(1.0 * NbRows / maxInCol) + 1;

  //char *y = new char[text[s].length() + 1];
  //std::strcpy(y, text[s].c_str());

  align = new TH2F(title.c_str(), "Ecal Function Parameters", NbColumns, 0, NbColumns, gridRows, 0, gridRows);

  double row = gridRows - 0.5;
  double column = 1.5;
  int cnt = 0;

  for (int i = 0; i < gridRows; i++) {
    align->Fill(0.5, gridRows - i - 0.5, i + 1);
  }

  for (std::vector<float>::const_iterator it = m_params.begin(); it != m_params.end(); it++) {
    if ((*it) == 0.0f)
      continue;
    align->Fill(column, row, *it);

    cnt++;
    column = floor(1.0 * cnt / maxInCol) + 1.5;
    row = (row == 0.5 ? (gridRows - 0.5) : row - 1);
  }
}