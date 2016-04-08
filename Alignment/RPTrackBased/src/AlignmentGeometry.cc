/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/RPTrackBased/interface/AlignmentGeometry.h"

using namespace std;
using namespace edm;

unsigned int AlignmentGeometry::MatrixIndexToDetId(unsigned int mi) const
{
  const_iterator it = FindByMatrixIndex(mi);
  if (it != end())
    return it->first;
  else {
    LogProblem("AlignmentGeometry") << ">> AlignmentGeometry::MatrixIndexToDetId > No detector corresponds to matrix index "
      << mi << ".";
    return 0;
  }
}

//----------------------------------------------------------------------------------------------------

AlignmentGeometry::const_iterator AlignmentGeometry::FindByMatrixIndex(unsigned int mi) const
{
  for (const_iterator it = begin(); it != end(); ++it)
    if (it->second.matrixIndex == mi)
      return it;
  return end();
}

//----------------------------------------------------------------------------------------------------

AlignmentGeometry::const_iterator AlignmentGeometry::FindFirstByRPMatrixIndex(unsigned int mi) const
{
  for (const_iterator it = begin(); it != end(); ++it)
    if (it->second.rpMatrixIndex == mi)
      return it;
  return end();
}

//----------------------------------------------------------------------------------------------------

void AlignmentGeometry::Print() const
{
  for (const_iterator it = begin(); it != end(); ++it) {
    const DetGeometry &d = it->second;
    printf("%u\t%+E\t%+E\t%+E\t%+E\t%+E\n", it->first, d.z, d.dx, d.dy, d.sx, d.sy);
  }
}

//----------------------------------------------------------------------------------------------------

void AlignmentGeometry::LoadFromFile(const std::string filename)
{
  clear();

  FILE *f = fopen(filename.c_str(), "r");
  if (!f)
    throw cms::Exception("AlignmentGeometry::LoadFromFile") << "File `" << filename << "' can not be opened." << endl;

  while (!feof(f)) {
    unsigned int id;
    float x, y, z, dx, dy;

    int res = fscanf(f, "%u%E%E%E%E%E", &id, &x, &y, &z, &dx, &dy);

    if (res == 6) {
      unsigned int rpNum = (id / 10) % 10;
      unsigned int detNum = id % 10;
      bool isU = (detNum % 2 != 0);
      if (rpNum == 2 || rpNum == 3)
        isU = !isU;
      Insert(id, DetGeometry(z, dx, dy, x, y, isU));      
    } else 
      if (!feof(f))
        throw cms::Exception("AlignmentGeometry::LoadFromFile") << "Cannot parse file `" << filename
          << "'. The format is probably wrong." << endl;
  }

  fclose(f);
}

