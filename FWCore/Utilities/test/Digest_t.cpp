#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cassert>
#include <iostream>

using cms::Digest;
using cms::MD5Result;

void testGivenString(std::string const& s) {
  Digest dig1(s);
  MD5Result r1 = dig1.digest();

  Digest dig2;
  dig2.append(s);
  MD5Result r2 = dig2.digest();
  assert(r1 == r2);

  // The result should be valid *iff* s is non-empty.
  assert(r1.isValid() == !s.empty() );
  assert(r1.toString().size() == 32);
  assert(r1.compactForm().size() == 16);
}

void testConversions() {
  std::string data("aldjfakl\tsdjf34234 \najdf");
  Digest dig(data);
  MD5Result r1 = dig.digest();
  assert(r1.isValid());
  std::string hexy = r1.toString();
  assert(hexy.size() == 32);
  MD5Result r2;
  r2.fromHexifiedString(hexy);
  assert(r1 == r2);
  assert(r1.toString() == r2.toString());
  assert(r1.compactForm() == r2.compactForm());

  //check the MD5Result lookup table
  MD5Result lookup;
  MD5Result fromHex;
  for(unsigned int i=0; i<256; ++i) {
    for(unsigned int j=0; j<16; ++j) {
      lookup.bytes[j]=static_cast<char>(i);
      fromHex.fromHexifiedString(lookup.toString());
      assert(lookup == fromHex);
      assert(lookup.toString() == fromHex.toString());
      assert(lookup.compactForm() == fromHex.compactForm());
    }
  }
}

void testEmptyString() {
  std::string e;
  testGivenString(e);

  Digest dig1;
  MD5Result r1 = dig1.digest();

  MD5Result r2;
  assert(r1 == r2);

  assert(!r1.isValid());
}

int main() {
  Digest dig1;
  dig1.append("hello");
  Digest dig2("hello");

  MD5Result r1 = dig1.digest();
  MD5Result r2 = dig2.digest();

  assert(r1 == r2);
  assert(!(r1 < r2));
  assert(!(r2 < r1));

  assert(r1.toString().size() == 32);

  testGivenString("a");
  testGivenString("{ }");
  testGivenString("abc 123 abc");
  testEmptyString();
  try {
    testConversions();
  }
  catch(cms::Exception const& e) {
    std::cerr << e.explainSelf() << std::endl; 
    return 1;
  }
  return 0;
}
