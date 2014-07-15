#include "CondFormats/DTObjects/interface/DTBufferTree.h"

#include <cassert>
#include <iostream>
#include <memory>

int main() {

  std::cout << "testDTBufferTree\n";

  DTBufferTree<int, int> buf;
  DTBufferTree<int, int> const * pBuf = &buf;
  std::vector<int> keys;

  int i;
  assert(buf.find(keys.end(), keys.end(), i) == 0);
  assert(i == 0);
  keys.push_back(1);
  assert(buf.find(keys.begin(), keys.end(), i) == 1);
  assert(i == 0);
  assert(buf.find(1, i) == 1);
  assert(i == 0);

  assert(pBuf->find(keys.end(), keys.end(), i) == 0);
  assert(i == 0);
  assert(pBuf->find(keys.begin(), keys.end(), i) == 1);
  assert(i == 0);
  assert(pBuf->find(1, i) == 1);
  assert(i == 0);

  keys.push_back(2);
  keys.push_back(3);
  assert(buf.insert(keys.begin(), keys.end(), 11) == 0);
  assert(buf.find(keys.begin(), keys.end(), i) == 0);
  assert(i == 11);
  assert(pBuf->find(keys.begin(), keys.end(), i) == 0);
  assert(i == 11);

  assert(buf.insert(keys.begin(), keys.end(), 12) == 1);
  assert(buf.find(keys.begin(), keys.end(), i) == 0);
  assert(i == 12);
  assert(pBuf->find(keys.begin(), keys.end(), i) == 0);
  assert(i == 12);

  assert(buf.insert(100, 111) == 0);
  assert(buf.find(100, i) == 0);
  assert(i == 111);
  assert(pBuf->find(100, i) == 0);
  assert(i == 111);

  assert(buf.insert(100, 112) == 1);
  assert(buf.find(100, i) == 0);
  assert(i == 112);
  assert(pBuf->find(100, i) == 0);
  assert(i == 112);

  assert(buf.insert(keys.end(), keys.end(), 1000) == 1);
  assert(buf.find(keys.end(), keys.end(), i) == 0);
  assert(i == 1000);
  assert(pBuf->find(keys.end(), keys.end(), i) == 0);
  assert(i == 1000);

  buf.clear();
  assert(buf.find(keys.end(), keys.end(), i) == 0);
  assert(i == 0);
  assert(pBuf->find(keys.end(), keys.end(), i) == 0);
  assert(i == 0);
  assert(buf.find(keys.begin(), keys.end(), i) == 1);
  assert(i == 0);
  assert(pBuf->find(keys.begin(), keys.end(), i) == 1);
  assert(i == 0);

  assert(buf.insert(keys.begin(), keys.end(), 7) == 0);
  assert(buf.find(keys.begin(), keys.end(), i) == 0);
  assert(i == 7);

  // ********************************************
  // Repeat with unique_ptr template parameter
  // Running this under valgrind checks for leaks
  // (which was done when the test was created)

  DTBufferTreeUniquePtr buf2;
  DTBufferTreeUniquePtr const * pBuf2 = &buf2;

  keys.clear();
  std::vector<int> v;
  std::vector<int> * ptr = &v;
  std::vector<int> const* cptr = &v;
  assert(buf2.find(keys.end(), keys.end(), ptr) == 0);
  assert(ptr == nullptr);
  keys.push_back(1);
  assert(buf2.find(keys.begin(), keys.end(), ptr) == 1);
  assert(ptr == nullptr);
  assert(buf2.find(1, ptr) == 1);
  assert(ptr == nullptr);

  assert(pBuf2->find(keys.end(), keys.end(), cptr) == 0);
  assert(cptr == nullptr);
  assert(pBuf2->find(keys.begin(), keys.end(), cptr) == 1);
  assert(cptr == nullptr);
  assert(pBuf2->find(1, cptr) == 1);
  assert(cptr == nullptr);

  keys.push_back(2);
  keys.push_back(3);
  std::unique_ptr<std::vector<int> > uptr(new std::vector<int>);
  uptr->push_back(101);
  uptr->push_back(102);
  assert(buf2.insert(keys.begin(), keys.end(), std::move(uptr)) == 0);
  assert(buf2.find(keys.begin(), keys.end(), ptr) == 0);
  assert(ptr->at(0) == 101 && ptr->at(1) == 102);
  ptr->push_back(103);
  assert(pBuf2->find(keys.begin(), keys.end(), cptr) == 0);
  assert(cptr->at(0) == 101 && cptr->at(1) == 102 && cptr->at(2) == 103);


  std::unique_ptr<std::vector<int> > uptr2(new std::vector<int>);
  uptr2->push_back(21);
  assert(buf2.insert(keys.begin(), keys.end(), std::move(uptr2)) == 1);
  assert(buf2.find(keys.begin(), keys.end(), ptr) == 0);
  assert(ptr->at(0) == 21);
  assert(pBuf2->find(keys.begin(), keys.end(), cptr) == 0);
  assert(cptr->at(0) == 21);

  std::unique_ptr<std::vector<int> > uptr3(new std::vector<int>);
  uptr3->push_back(31);
  assert(buf2.insert(100, std::move(uptr3)) == 0);
  assert(buf2.find(100, ptr) == 0);
  assert(ptr->at(0) == 31);
  assert(pBuf2->find(100, cptr) == 0);
  assert(cptr->at(0) == 31);

  std::unique_ptr<std::vector<int> > uptr4(new std::vector<int>);
  uptr4->push_back(41);
  assert(buf2.insert(100, std::move(uptr4)) == 1);
  assert(buf2.find(100, ptr) == 0);
  assert(ptr->at(0) == 41);
  assert(pBuf2->find(100, cptr) == 0);
  assert(cptr->at(0) == 41);

  std::unique_ptr<std::vector<int> > uptr5(new std::vector<int>);
  uptr5->push_back(51);
  assert(buf2.insert(keys.end(), keys.end(), std::move(uptr5)) == 1);
  assert(buf2.find(keys.end(), keys.end(), ptr) == 0);
  assert(ptr->at(0) == 51);
  assert(pBuf2->find(keys.end(), keys.end(), cptr) == 0);
  assert(cptr->at(0) == 51);

  buf2.clear();
  assert(buf2.find(keys.end(), keys.end(), ptr) == 0);
  assert(ptr == nullptr);
  assert(pBuf2->find(keys.end(), keys.end(), cptr) == 0);
  assert(cptr == nullptr);
  assert(buf2.find(keys.begin(), keys.end(), ptr) == 1);
  assert(ptr == nullptr);
  assert(pBuf2->find(keys.begin(), keys.end(), cptr) == 1);
  assert(cptr == nullptr);

  std::unique_ptr<std::vector<int> > uptr6(new std::vector<int>);
  uptr6->push_back(61);
  assert(buf2.insert(keys.begin(), keys.end(), std::move(uptr6)) == 0);
  assert(buf2.find(keys.begin(), keys.end(), ptr) == 0);
  assert(ptr->at(0) == 61);
}
