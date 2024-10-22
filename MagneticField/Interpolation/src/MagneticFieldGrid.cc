// include header for MagneticFieldGrid (regular + extension for some trapezoids)
#include "MagneticFieldGrid.h"
#include "MagneticField/Interpolation/interface/binary_ifstream.h"
#include <cassert>

using namespace std;

void MagneticFieldGrid::load(const string &name) {
  magneticfield::interpolation::binary_ifstream inFile(name);
  inFile >> GridType;
  // reading the header
  switch (GridType) {
    case 1:
      inFile >> NumberOfPoints[0] >> NumberOfPoints[1] >> NumberOfPoints[2];
      inFile >> ReferencePoint[0] >> ReferencePoint[1] >> ReferencePoint[2];
      inFile >> BasicDistance0[0] >> BasicDistance0[1] >> BasicDistance0[2];
      break;
    case 2:
      inFile >> NumberOfPoints[0] >> NumberOfPoints[1] >> NumberOfPoints[2];
      inFile >> ReferencePoint[0] >> ReferencePoint[1] >> ReferencePoint[2];
      inFile >> BasicDistance0[0] >> BasicDistance0[1] >> BasicDistance0[2];
      inFile >> BasicDistance1[0][0] >> BasicDistance1[1][0] >> BasicDistance1[2][0];
      inFile >> BasicDistance1[0][1] >> BasicDistance1[1][1] >> BasicDistance1[2][1];
      inFile >> BasicDistance1[0][2] >> BasicDistance1[1][2] >> BasicDistance1[2][2];
      inFile >> BasicDistance2[0][0] >> BasicDistance2[1][0] >> BasicDistance2[2][0];
      inFile >> BasicDistance2[0][1] >> BasicDistance2[1][1] >> BasicDistance2[2][1];
      inFile >> BasicDistance2[0][2] >> BasicDistance2[1][2] >> BasicDistance2[2][2];
      inFile >> EasyCoordinate[0] >> EasyCoordinate[1] >> EasyCoordinate[2];
      break;
    case 3:
      inFile >> NumberOfPoints[0] >> NumberOfPoints[1] >> NumberOfPoints[2];
      inFile >> ReferencePoint[0] >> ReferencePoint[1] >> ReferencePoint[2];
      inFile >> BasicDistance0[0] >> BasicDistance0[1] >> BasicDistance0[2];
      break;
    case 4:
      inFile >> NumberOfPoints[0] >> NumberOfPoints[1] >> NumberOfPoints[2];
      inFile >> ReferencePoint[0] >> ReferencePoint[1] >> ReferencePoint[2];
      inFile >> BasicDistance0[0] >> BasicDistance0[1] >> BasicDistance0[2];
      inFile >> BasicDistance1[0][0] >> BasicDistance1[1][0] >> BasicDistance1[2][0];
      inFile >> BasicDistance1[0][1] >> BasicDistance1[1][1] >> BasicDistance1[2][1];
      inFile >> BasicDistance1[0][2] >> BasicDistance1[1][2] >> BasicDistance1[2][2];
      inFile >> BasicDistance2[0][0] >> BasicDistance2[1][0] >> BasicDistance2[2][0];
      inFile >> BasicDistance2[0][1] >> BasicDistance2[1][1] >> BasicDistance2[2][1];
      inFile >> BasicDistance2[0][2] >> BasicDistance2[1][2] >> BasicDistance2[2][2];
      inFile >> EasyCoordinate[0] >> EasyCoordinate[1] >> EasyCoordinate[2];
      break;
    case 5:
      inFile >> NumberOfPoints[0] >> NumberOfPoints[1] >> NumberOfPoints[2];
      inFile >> ReferencePoint[0] >> ReferencePoint[1] >> ReferencePoint[2];
      inFile >> BasicDistance0[0] >> BasicDistance0[1] >> BasicDistance0[2];
      inFile >> RParAsFunOfPhi[0] >> RParAsFunOfPhi[1] >> RParAsFunOfPhi[2] >> RParAsFunOfPhi[3];
      break;
    default:
      assert(0);  //this is a bug
  }
  //reading the field
  float Bx, By, Bz;
  BVector FieldEntry;
  int nLines = NumberOfPoints[0] * NumberOfPoints[1] * NumberOfPoints[2];
  FieldValues.reserve(nLines);

  for (int iLine = 0; iLine < nLines; ++iLine) {
    inFile >> Bx >> By >> Bz;
    FieldEntry.putB3(Bx, By, Bz);
    FieldValues.push_back(FieldEntry);
  }
  // check completeness and close file
  string lastEntry;
  inFile >> lastEntry;
  inFile.close();
  if (lastEntry != "complete") {
    GridType = 0;
    cout << "error during file reading: file is not complete" << endl;
  }
  return;
}

int MagneticFieldGrid::gridType() {
  int type = GridType;
  bool text = false;
  if (text) {
    if (type == 0)
      cout << "  grid type = " << type << "  -->  not determined" << endl;
    if (type == 1)
      cout << "  grid type = " << type << "  -->  (x,y,z) cube" << endl;
    if (type == 2)
      cout << "  grid type = " << type << "  -->  (x,y,z) trapezoid" << endl;
    if (type == 3)
      cout << "  grid type = " << type << "  -->  (r,phi,z) cube" << endl;
    if (type == 4)
      cout << "  grid type = " << type << "  -->  (r,phi,z) trapezoid" << endl;
    if (type == 5)
      cout << "  grid type = " << type << "  -->  (r,phi,z) 1/sin(phi)" << endl;
  }
  return type;
}

void MagneticFieldGrid::interpolateAtPoint(double X1, double X2, double X3, float &Bx, float &By, float &Bz) {
  double dB[3] = {0., 0., 0.};
  // define interpolation object
  VectorFieldInterpolation MagInterpol;
  // calculate indices for "CellPoint000"
  int index[3];
  putCoordGetInd(X1, X2, X3, index[0], index[1], index[2]);
  int index0[3] = {0, 0, 0};
  int index1[3] = {0, 0, 0};
  for (int i = 0; i < 3; ++i) {
    if (NumberOfPoints[i] > 1) {
      index0[i] = max(0, index[i]);
      if (index0[i] > NumberOfPoints[i] - 2)
        index0[i] = NumberOfPoints[i] - 2;
      index1[i] = max(1, index[i] + 1);
      if (index1[i] > NumberOfPoints[i] - 1)
        index1[i] = NumberOfPoints[i] - 1;
    }
  }
  double tmpX[3];
  float tmpB[3];
  // define the corners of interpolation volume
  // FIXME: should not unpack the arrays to then repack them as first thing.
  putIndicesGetB(index0[0], index0[1], index0[2], tmpB[0], tmpB[1], tmpB[2]);
  putIndGetCoord(index0[0], index0[1], index0[2], tmpX[0], tmpX[1], tmpX[2]);
  MagInterpol.defineCellPoint000(tmpX[0], tmpX[1], tmpX[2], double(tmpB[0]), double(tmpB[1]), double(tmpB[2]));
  putIndicesGetB(index1[0], index0[1], index0[2], tmpB[0], tmpB[1], tmpB[2]);
  putIndGetCoord(index1[0], index0[1], index0[2], tmpX[0], tmpX[1], tmpX[2]);
  MagInterpol.defineCellPoint100(tmpX[0], tmpX[1], tmpX[2], double(tmpB[0]), double(tmpB[1]), double(tmpB[2]));
  putIndicesGetB(index0[0], index1[1], index0[2], tmpB[0], tmpB[1], tmpB[2]);
  putIndGetCoord(index0[0], index1[1], index0[2], tmpX[0], tmpX[1], tmpX[2]);
  MagInterpol.defineCellPoint010(tmpX[0], tmpX[1], tmpX[2], double(tmpB[0]), double(tmpB[1]), double(tmpB[2]));
  putIndicesGetB(index1[0], index1[1], index0[2], tmpB[0], tmpB[1], tmpB[2]);
  putIndGetCoord(index1[0], index1[1], index0[2], tmpX[0], tmpX[1], tmpX[2]);
  MagInterpol.defineCellPoint110(tmpX[0], tmpX[1], tmpX[2], double(tmpB[0]), double(tmpB[1]), double(tmpB[2]));
  putIndicesGetB(index0[0], index0[1], index1[2], tmpB[0], tmpB[1], tmpB[2]);
  putIndGetCoord(index0[0], index0[1], index1[2], tmpX[0], tmpX[1], tmpX[2]);
  MagInterpol.defineCellPoint001(tmpX[0], tmpX[1], tmpX[2], double(tmpB[0]), double(tmpB[1]), double(tmpB[2]));
  putIndicesGetB(index1[0], index0[1], index1[2], tmpB[0], tmpB[1], tmpB[2]);
  putIndGetCoord(index1[0], index0[1], index1[2], tmpX[0], tmpX[1], tmpX[2]);
  MagInterpol.defineCellPoint101(tmpX[0], tmpX[1], tmpX[2], double(tmpB[0]), double(tmpB[1]), double(tmpB[2]));
  putIndicesGetB(index0[0], index1[1], index1[2], tmpB[0], tmpB[1], tmpB[2]);
  putIndGetCoord(index0[0], index1[1], index1[2], tmpX[0], tmpX[1], tmpX[2]);
  MagInterpol.defineCellPoint011(tmpX[0], tmpX[1], tmpX[2], double(tmpB[0]), double(tmpB[1]), double(tmpB[2]));
  putIndicesGetB(index1[0], index1[1], index1[2], tmpB[0], tmpB[1], tmpB[2]);
  putIndGetCoord(index1[0], index1[1], index1[2], tmpX[0], tmpX[1], tmpX[2]);
  MagInterpol.defineCellPoint111(tmpX[0], tmpX[1], tmpX[2], double(tmpB[0]), double(tmpB[1]), double(tmpB[2]));
  // interpolate
  MagInterpol.putSCoordGetVField(X1, X2, X3, dB[0], dB[1], dB[2]);
  Bx = float(dB[0]);
  By = float(dB[1]);
  Bz = float(dB[2]);
  return;
}

// FIXME: Signature should be:
//
//     void MagneticFieldGrid::putCoordGetInd(double *pos, int *index)
//
void MagneticFieldGrid::putCoordGetInd(double X1, double X2, double X3, int &Index1, int &Index2, int &Index3) {
  double pnt[3] = {X1, X2, X3};
  int index[3];
  switch (GridType) {
    case 1: {
      for (int i = 0; i < 3; ++i) {
        index[i] = int((pnt[i] - ReferencePoint[i]) / BasicDistance0[i]);
      }
      break;
    }
    case 2: {
      // FIXME: Should use else!
      for (int i = 0; i < 3; ++i) {
        if (EasyCoordinate[i]) {
          index[i] = int((pnt[i] - ReferencePoint[i]) / BasicDistance0[i]);
        } else
          index[i] = 0;  //computed below
      }
      for (int i = 0; i < 3; ++i) {
        if (!EasyCoordinate[i]) {
          double stepSize = BasicDistance0[i];
          double offset = 0.0;
          for (int j = 0; j < 3; ++j) {
            stepSize += BasicDistance1[i][j] * index[j];
            offset += BasicDistance2[i][j] * index[j];
          }
          index[i] = int((pnt[i] - (ReferencePoint[i] + offset)) / stepSize);
        }
      }
      break;
    }
    case 3: {
      for (int i = 0; i < 3; ++i) {
        index[i] = int((pnt[i] - ReferencePoint[i]) / BasicDistance0[i]);
      }
      break;
    }
    case 4: {
      // FIXME: should use else!
      for (int i = 0; i < 3; ++i) {
        if (EasyCoordinate[i]) {
          index[i] = int((pnt[i] - ReferencePoint[i]) / BasicDistance0[i]);
        } else
          index[i] = 0;  //computed below
      }
      for (int i = 0; i < 3; ++i) {
        if (!EasyCoordinate[i]) {
          double stepSize = BasicDistance0[i];
          double offset = 0.0;
          for (int j = 0; j < 3; ++j) {
            stepSize += BasicDistance1[i][j] * index[j];
            offset += BasicDistance2[i][j] * index[j];
          }
          index[i] = int((pnt[i] - (ReferencePoint[i] + offset)) / stepSize);
        }
      }
      break;
    }
    case 5: {
      double sinPhi = sin(pnt[1]);
      double stepSize = RParAsFunOfPhi[0] + RParAsFunOfPhi[1] / sinPhi - RParAsFunOfPhi[2] - RParAsFunOfPhi[3] / sinPhi;
      stepSize = stepSize / (NumberOfPoints[0] - 1);
      double startingPoint = RParAsFunOfPhi[2] + RParAsFunOfPhi[3] / sinPhi;
      index[0] = int((pnt[0] - startingPoint) / stepSize);
      index[1] = int((pnt[1] - ReferencePoint[1]) / BasicDistance0[1]);
      index[2] = int((pnt[2] - ReferencePoint[2]) / BasicDistance0[2]);
      break;
    }
    default:
      assert(0);  //shouldn't be here
  }
  Index1 = index[0];
  Index2 = index[1];
  Index3 = index[2];
  return;
}

void MagneticFieldGrid::putIndicesGetB(int Index1, int Index2, int Index3, float &Bx, float &By, float &Bz) {
  BVector FieldEntry;
  FieldEntry = FieldValues.operator[](lineNumber(Index1, Index2, Index3));
  Bx = FieldEntry.bx();
  By = FieldEntry.by();
  Bz = FieldEntry.bz();
  return;
}

// FIXME: same as above.
void MagneticFieldGrid::putIndGetCoord(int Index1, int Index2, int Index3, double &X1, double &X2, double &X3) {
  int index[3] = {Index1, Index2, Index3};
  double pnt[3];
  switch (GridType) {
    case 1: {
      for (int i = 0; i < 3; ++i) {
        pnt[i] = ReferencePoint[i] + BasicDistance0[i] * index[i];
      }
      break;
    }
    case 2: {
      for (int i = 0; i < 3; ++i) {
        if (EasyCoordinate[i]) {
          pnt[i] = ReferencePoint[i] + BasicDistance0[i] * index[i];
        } else {
          double stepSize = BasicDistance0[i];
          double offset = 0.0;
          for (int j = 0; j < 3; ++j) {
            stepSize += BasicDistance1[i][j] * index[j];
            offset += BasicDistance2[i][j] * index[j];
          }
          pnt[i] = ReferencePoint[i] + offset + stepSize * index[i];
        }
      }
      break;
    }
    case 3: {
      for (int i = 0; i < 3; ++i) {
        pnt[i] = ReferencePoint[i] + BasicDistance0[i] * index[i];
      }
      break;
    }
    case 4: {
      for (int i = 0; i < 3; ++i) {
        if (EasyCoordinate[i]) {
          pnt[i] = ReferencePoint[i] + BasicDistance0[i] * index[i];
        } else {
          double stepSize = BasicDistance0[i];
          double offset = 0.0;
          for (int j = 0; j < 3; ++j) {
            stepSize += BasicDistance1[i][j] * index[j];
            offset += BasicDistance2[i][j] * index[j];
          }
          pnt[i] = ReferencePoint[i] + offset + stepSize * index[i];
        }
      }
      break;
    }
    case 5: {
      pnt[2] = ReferencePoint[2] + BasicDistance0[2] * index[2];
      pnt[1] = ReferencePoint[1] + BasicDistance0[1] * index[1];
      double sinPhi = sin(pnt[1]);
      double stepSize = RParAsFunOfPhi[0] + RParAsFunOfPhi[1] / sinPhi - RParAsFunOfPhi[2] - RParAsFunOfPhi[3] / sinPhi;
      stepSize = stepSize / (NumberOfPoints[0] - 1);
      double startingPoint = RParAsFunOfPhi[2] + RParAsFunOfPhi[3] / sinPhi;
      pnt[0] = startingPoint + stepSize * index[0];
      break;
    }
    default:
      assert(0);  //bug if make it here
  }
  X1 = pnt[0];
  X2 = pnt[1];
  X3 = pnt[2];
  return;
}

int MagneticFieldGrid::lineNumber(int Index1, int Index2, int Index3) {
  return Index1 * NumberOfPoints[1] * NumberOfPoints[2] + Index2 * NumberOfPoints[2] + Index3;
}

void MagneticFieldGrid::BVector::putB3(float Bx, float By, float Bz) {
  B3[0] = Bx;
  B3[1] = By;
  B3[2] = Bz;
  return;
}

float MagneticFieldGrid::BVector::bx() { return B3[0]; }

float MagneticFieldGrid::BVector::by() { return B3[1]; }

float MagneticFieldGrid::BVector::bz() { return B3[2]; }
