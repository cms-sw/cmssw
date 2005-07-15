#include "FWCore/Utilities/interface/Digest.h"

#include <iostream>
#include <string>

// ------------------------------------------------------------
// The following are the strings, and their MD5 hexdigests, as
// expressed in RFC 1321, which defines the MD5 algorithm.
// ------------------------------------------------------------

const char* const table[7*2] = {
  "", "d41d8cd98f00b204e9800998ecf8427e",
  "a", "0cc175b9c0f1b6a831c399e269772661",
  "abc", "900150983cd24fb0d6963f7d28e17f72",
  "message digest", "f96b697d7cb7938d525a2f31aaf161d0",
  "abcdefghijklmnopqrstuvwxyz", "c3fcd3d76192e4007dfb496cca67e13b",
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
  "d174ab98d277d9f5a5611c2c9f419d9f",
  "12345678901234567890123456789012345678901234567890123456789012345678901234567890", "57edf4a22be3c955ac49da2e2107b67a"
};

int main()
{
  int numberOfFailures(0);
  for (int i = 0; i < 7*2 ; i += 2)
    {
      cms::Digest dig(table[i]);
      std::string hex = dig.digest().toString();
      std::string expected = table[i+1];
      if (hex != expected)
	{
	  ++numberOfFailures;
	  std::cerr << "--------------------------------\n";
	  std::cerr << "Expected: " << expected << '\n';
	  std::cerr << "Result:   " << hex << '\n';
	}
    }
  return numberOfFailures;
}
