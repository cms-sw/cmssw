#include "DataFormats/NanoAOD/interface/FlatTable.h"

int nanoaod::FlatTable::columnIndex(const std::string& name) const {
  for (unsigned int i = 0, n = columns_.size(); i < n; ++i) {
    if (columns_[i].name == name)
      return i;
  }
  return -1;
}

void nanoaod::FlatTable::addExtension(const nanoaod::FlatTable& other) {
  if (extension() || !other.extension() || name() != other.name() || size() != other.size())
    throw cms::Exception("LogicError", "Mismatch in adding extension");
  for (unsigned int i = 0, n = other.nColumns(); i < n; ++i) {
    switch (other.columnType(i)) {
      case FloatColumn:
        addColumn<float>(other.columnName(i), other.columnData<float>(i), other.columnDoc(i));
        break;
      case IntColumn:
        addColumn<int>(other.columnName(i), other.columnData<int>(i), other.columnDoc(i));
        break;
      case BoolColumn:
        addColumn<bool>(other.columnName(i), other.columnData<uint8_t>(i), other.columnDoc(i));
        break;
      case UInt8Column:
        addColumn<uint8_t>(other.columnName(i), other.columnData<uint8_t>(i), other.columnDoc(i));
        break;
    }
  }
}

double nanoaod::FlatTable::getAnyValue(unsigned int row, unsigned int column) const {
  if (column >= nColumns())
    throw cms::Exception("LogicError", "Invalid column");
  switch (columnType(column)) {
    case FloatColumn:
      return *(beginData<float>(column) + row);
    case IntColumn:
      return *(beginData<int>(column) + row);
    case BoolColumn:
      return *(beginData<uint8_t>(column) + row);
    case UInt8Column:
      return *(beginData<uint8_t>(column) + row);
  }
  throw cms::Exception("LogicError", "Unsupported type");
}
