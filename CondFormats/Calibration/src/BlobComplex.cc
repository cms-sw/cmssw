#include <iostream>

#include "CondFormats/Calibration/interface/BlobComplex.h"

void BlobComplexData::fill(unsigned int &serial)
{
  a = ++serial;
  b = ++serial;
  for(unsigned int i = 0; i < 10; i++)
    values.push_back(++serial);
}

void BlobComplexData::print() const
{
  std::cout << "        a = " << a << std::endl;
  std::cout << "        b = " << b << std::endl;
  for(std::vector<unsigned int>::const_iterator iter = values.begin();
      iter != values.end(); iter++)
    std::cout << "        value[" << (iter - values.begin()) << "] = "
              << *iter << std::endl;
}

bool BlobComplexData::operator == (const BlobComplexData &rhs) const
{
  if (a != rhs.a) return false;
  if (b != rhs.b) return false;
  if (values.size() != rhs.values.size()) return false;
  std::vector<unsigned int>::const_iterator iter1 = values.begin();
  std::vector<unsigned int>::const_iterator iter2 = rhs.values.begin();
  while(iter1 != values.end())
    if (*iter1++ != *iter2++) return false;
  return true;
}

void BlobComplexContent::fill(unsigned int &serial)
{
  data1.first.fill(serial);
  data1.second = ++serial;
  data2.first.fill(serial);
  data2.second = ++serial;
  data3.first.fill(serial);
  data3.second = ++serial;
}

static void printBlobComplexContentData(const BlobComplexContent::Data &data)
{
  std::cout << "      first = " << std::endl;
  data.first.print();
  std::cout << "      second = " << data.second << std::endl;
}

void BlobComplexContent::print() const
{
  std::cout << "    data1 = " << std::endl;
  printBlobComplexContentData(data1);
  std::cout << "    data2 = " << std::endl;
  printBlobComplexContentData(data2);
  std::cout << "    data3 = " << std::endl;
  printBlobComplexContentData(data3);
}

bool BlobComplexContent::operator == (const BlobComplexContent &rhs) const
{
  if (data1.first != rhs.data1.first ||
      data1.second != rhs.data1.second) return false;
  if (data2.first != rhs.data2.first ||
      data2.second != rhs.data2.second) return false;
  if (data3.first != rhs.data3.first ||
      data3.second != rhs.data3.second) return false;
  return true;
}

void BlobComplexObjects::fill(unsigned int &serial)
{
  a = ++serial;
  b = ++serial;
  for(unsigned int i = 0; i < 3; i++) {
    content.push_back(BlobComplexContent());
    content.back().fill(serial);
  }
}

void BlobComplexObjects::print() const
{
  std::cout << "  a = " << a << std::endl;
  std::cout << "  b = " << b << std::endl;
  for(std::vector<BlobComplexContent>::const_iterator iter = content.begin();
      iter != content.end(); iter++) {
    std::cout << "  content[" << (iter - content.begin()) << "] =" << std::endl;
    iter->print();
  }
}

bool BlobComplexObjects::operator == (const BlobComplexObjects &rhs) const
{
  if (a != rhs.a) return false;
  if (b != rhs.b) return false;
  if (content.size() != rhs.content.size()) return false;
  std::vector<BlobComplexContent>::const_iterator iter1 = content.begin();
  std::vector<BlobComplexContent>::const_iterator iter2 = rhs.content.begin();
  while(iter1 != content.end())
    if (*iter1++ != *iter2++) return false;
  return true;
}

void BlobComplex::fill(unsigned int &serial)
{
  for(unsigned int i = 0; i < 3; i++) {
    objects.push_back(BlobComplexObjects());
    objects.back().fill(serial);
  }
}

void BlobComplex::print() const
{
  for(std::vector<BlobComplexObjects>::const_iterator iter = objects.begin();
      iter != objects.end(); iter++) {
    std::cout << "objects[" << (iter - objects.begin()) << "] =" << std::endl;
    iter->print();
  }
}

bool BlobComplex::operator == (const BlobComplex &rhs) const
{
  if (objects.size() != rhs.objects.size()) return false;
  std::vector<BlobComplexObjects>::const_iterator iter1 = objects.begin();
  std::vector<BlobComplexObjects>::const_iterator iter2 = rhs.objects.begin();
  while(iter1 != objects.end())
    if (*iter1++ != *iter2++) return false;
  return true;
}
