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
      case ColumnType::Float:
        addColumn<float>(other.columnName(i), other.columnData<float>(i), other.columnDoc(i));
        break;
      case ColumnType::Int:
        addColumn<int>(other.columnName(i), other.columnData<int>(i), other.columnDoc(i));
        break;
      case ColumnType::Bool:
        addColumn<bool>(other.columnName(i), other.columnData<bool>(i), other.columnDoc(i));
        break;
      case ColumnType::UInt8:
        addColumn<uint8_t>(other.columnName(i), other.columnData<uint8_t>(i), other.columnDoc(i));
        break;
      case ColumnType::UInt32:
        addColumn<uint32_t>(other.columnName(i), other.columnData<uint32_t>(i), other.columnDoc(i));
        break;
      case ColumnType::Double:
        addColumn<double>(other.columnName(i), other.columnData<double>(i), other.columnDoc(i));
        break;
      default:
        throw cms::Exception("LogicError", "Unsupported type");
    }
  }
}

double nanoaod::FlatTable::getAnyValue(unsigned int row, unsigned int column) const {
  if (column >= nColumns())
    throw cms::Exception("LogicError", "Invalid column");
  switch (columnType(column)) {
    case ColumnType::Float:
      return *(beginData<float>(column) + row);
    case ColumnType::Int:
      return *(beginData<int>(column) + row);
    case ColumnType::Bool:
      return *(beginData<bool>(column) + row);
    case ColumnType::UInt8:
      return *(beginData<uint8_t>(column) + row);
    case ColumnType::UInt32:
      return *(beginData<uint32_t>(column) + row);
    case ColumnType::Double:
      return *(beginData<double>(column) + row);
  }
  throw cms::Exception("LogicError", "Unsupported type");
}
