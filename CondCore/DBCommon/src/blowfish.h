/*
 * Author     :  Paul Kocher
 * E-mail     :  pck@netcom.com
 * Date       :  1997
 * Description:  C implementation of the Blowfish algorithm.
 */


#define INCLUDE_BLOWFISH_DEFINE_H

struct BCoptions{
  unsigned char remove;
  unsigned char standardout;
  unsigned char compression;
  unsigned char type;
  unsigned long origsize;
  unsigned char securedelete;
};

#define ENCRYPT 0
#define DECRYPT 1

#define endianBig ((unsigned char) 0x45)
#define endianLittle ((unsigned char) 0x54)

typedef unsigned int uInt32;

#ifdef WIN32 /* Win32 doesn't have random() or lstat */
#define random() rand()
#define initstate(x,y,z) srand(x)
#define lstat(x,y) stat(x,y)
#endif

#ifndef S_ISREG
#define S_ISREG(x) ( ((x)&S_IFMT)==S_IFREG )
#endif

#define MAXKEYBYTES 56          /* 448 bits */

struct BLOWFISH_CTX {
  uInt32 P[16 + 2];
  uInt32 S[4][256];
};

void Blowfish_Init(BLOWFISH_CTX *ctx, unsigned char *key, int keyLen);

void Blowfish_Encrypt(BLOWFISH_CTX *ctx, uInt32 *xl, uInt32
*xr);

void Blowfish_Decrypt(BLOWFISH_CTX *ctx, uInt32 *xl, uInt32
*xr);
