#ifndef CALIBRATION_INTERFACE_H
#define CALIBRATION_INTERFACE_H
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>
#include <utility>
#include <iostream>

/**

*/
template <class CategoryT, class CalibDataT>
class CalibrationInterface {
public:
  CalibrationInterface();
  ~CalibrationInterface();

  const CalibDataT* getCalibData(const typename CategoryT::Input& calibrationInput) const {
    return (getCalibData(getIndex(calibrationInput)));
  }

  CalibDataT* getCalibData(const typename CategoryT::Input& calibrationInput) {
    return (getCalibData(getIndex(calibrationInput)));
  }

  const CalibDataT* getCalibData(int index) const;
  CalibDataT* getCalibData(int index);

  const CategoryT* getCategoryDefinition(int index) const;

  int getIndex(const typename CategoryT::Input& calibrationInput) const;

  int addCategoryDefinition(const CategoryT& categoryDefinition);
  int addEntry(const CategoryT& categoryDefinition, const CalibDataT& data);

  void setCalibData(int index, const CalibDataT& data);

  const std::vector<std::pair<CategoryT, CalibDataT> >& categoriesWithData() const { return m_categoriesWithData; }
  int size() const { return m_categoriesWithData.size(); }

private:
  std::vector<std::pair<CategoryT, CalibDataT> > m_categoriesWithData;
};

template <class CategoryT, class CalibDataT>
CalibrationInterface<CategoryT, CalibDataT>::CalibrationInterface() {}

template <class CategoryT, class CalibDataT>
CalibrationInterface<CategoryT, CalibDataT>::~CalibrationInterface() {}

template <class CategoryT, class CalibDataT>
int CalibrationInterface<CategoryT, CalibDataT>::getIndex(const typename CategoryT::Input& calibrationInput) const {
  int i = 0;
  int found = -1;
  for (typename std::vector<std::pair<CategoryT, CalibDataT> >::const_iterator it = m_categoriesWithData.begin();
       it != m_categoriesWithData.end();
       it++) {
    if ((*it).first.match(calibrationInput)) {
      if (found >= 0) {
        edm::LogWarning("BTagCalibration") << "WARNING: OVERLAP in categories, using latest one";
      }

      found = i;
    }
    i++;
  }
  return found;
}
template <class CategoryT, class CalibDataT>
const CalibDataT* CalibrationInterface<CategoryT, CalibDataT>::getCalibData(int i) const {
  size_t ii = i;
  if (i >= 0 && ii < m_categoriesWithData.size())
    return &m_categoriesWithData[i].second;
  else
    return 0;
}

template <class CategoryT, class CalibDataT>
CalibDataT* CalibrationInterface<CategoryT, CalibDataT>::getCalibData(int i) {
  size_t ii = i;
  if (i >= 0 && ii < m_categoriesWithData.size())
    return &m_categoriesWithData[i].second;
  else
    return 0;
}

template <class CategoryT, class CalibDataT>
int CalibrationInterface<CategoryT, CalibDataT>::addCategoryDefinition(const CategoryT& categoryDefinition) {
  CalibDataT emptyData;
  return addEntry(categoryDefinition, emptyData);
}

template <class CategoryT, class CalibDataT>
int CalibrationInterface<CategoryT, CalibDataT>::addEntry(const CategoryT& categoryDefinition, const CalibDataT& data) {
  std::pair<CategoryT, CalibDataT> newEntry(categoryDefinition, data);
  m_categoriesWithData.push_back(newEntry);
  return m_categoriesWithData.size();
}

template <class CategoryT, class CalibDataT>
void CalibrationInterface<CategoryT, CalibDataT>::setCalibData(int index, const CalibDataT& data) {
  m_categoriesWithData[index].second = data;
}
#endif
