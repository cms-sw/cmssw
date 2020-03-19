#include "Alignment/SurveyAnalysis/interface/SurveyPxbImage.h"

#include <stdexcept>
#include <utility>
#include <sstream>
#include <vector>
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

void SurveyPxbImage::fill(std::istringstream &iss) {
  id_t id1, id2;
  value_t x0, y0;
  value_t x1, y1;
  value_t x2, y2;
  value_t x3, y3;
  value_t sx, sy;
  bool rotflag;
  if (!(iss >> id1 >> x0 >> y0 >> x1 >> y1 >> id2 >> x2 >> y2 >> x3 >> y3 >> sy >> sx >> rotflag).fail()) {
    idPair_.first = id1;
    idPair_.second = id2;
    if (!rotflag) {
      measurementVec_.push_back(coord_t(x0, -y0));
      measurementVec_.push_back(coord_t(x1, -y1));
      measurementVec_.push_back(coord_t(x2, -y2));
      measurementVec_.push_back(coord_t(x3, -y3));
    } else {
      measurementVec_.push_back(coord_t(-x0, y0));
      measurementVec_.push_back(coord_t(-x1, y1));
      measurementVec_.push_back(coord_t(-x2, y2));
      measurementVec_.push_back(coord_t(-x3, y3));
    }
    sigma_x_ = sx;
    sigma_y_ = sy;
    isRotated_ = rotflag;
    isValidFlag_ = true;
  } else
    isValidFlag_ = false;
}

const SurveyPxbImage::coord_t SurveyPxbImage::getCoord(count_t m) {
  if (m > 0 && m < 5)
    return measurementVec_[m - 1];
  else
    throw std::out_of_range("Attempt to access an inexistent measurement");
}
