#ifndef HoughStruct_def_
#define HoughStruct_def_
#include <stdint.h>
#include <vector>
typedef struct {
  int32_t id;
  double phi;
  double pt;
  double eta;
  double z0;
  double rho0;
  uint32_t nstubs;
  uint32_t maxstubs;
  uint32_t matches;
  uint32_t nhits;
  bool valid;
  uint32_t id_ass;
  float pterr,phierr;
  float r,theta;
  uint16_t tag;
  float chi2;
  float chi2z;
  std::vector<uint32_t> layers;
} mctrack_t;


typedef struct
{
  uint32_t id;
  float x,y,z;
  float xp,yp,r2,r;
  int32_t tp;
  uint16_t layer;
} stub_t;

#define STUBID(layer,ladder,module,segment,strip) (((layer&0x7f)<<25)|((ladder&0x7f)<<18)|((module&0x7f)<<11)| ((segment&0x1)<<10)| (strip&0x3ff))
#define LAYER(ID) ((ID>>25)&0x7F)
#define LADDER(ID) ((ID>>18)&0x7F)
#define MODULE(ID) ((ID>>11)&0x7F)
#define SEGMENT(ID) ((ID>>10)&0x1)
#define STRIP(ID) ((I&0x3FF))


typedef struct
{
  std::vector<uint32_t> stubs_id;
 
} pattern_t;
#endif
