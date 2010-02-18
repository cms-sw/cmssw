#include "Alignment/SurveyAnalysis/interface/SurveyPxbImage.h"

#include <stdexcept>
#include <utility>
#include <sstream>
#include <vector>
#include "DataFormats/GeometryVector/interface/LocalPoint.h"


SurveyPxbImage::SurveyPxbImage(std::istringstream &iss) : isValidFlag_(false)
{
	fill(iss);
}

void SurveyPxbImage::fill(std::istringstream &iss)
{
  id_t id1, id2;
  value_t u11, v11, u21, v21;
  value_t u12, v12, u22, v22;
  value_t sv, su;
  if(! (iss >> id1 >> v11 >> u11 >> v21 >> u21
        >> id2 >> v12 >> u12 >> v22 >> u22
        >> sv >> su).fail())
    {
      idPair_.first = id1;
      idPair_.second = id2;
      // Flip sign of u to change into CMS coord system
      const coord_t c4(u22,-v22);
      measurementVec_.push_back(c4);
      const coord_t c3(u12,-v12);
      measurementVec_.push_back(c3);
      const coord_t c2(u11,-v11);
      measurementVec_.push_back(c2);
      const coord_t c1(u21,-v21);
      measurementVec_.push_back(c1);
      sigma_v_ = sv;
      sigma_u_ = su;
      isValidFlag_ = true;
    }
  else
    isValidFlag_ = false;
}

const SurveyPxbImage::coord_t SurveyPxbImage::getCoord(count_t m)
{
  if (m>0 && m<5)
    return measurementVec_[m-1];
  else
    throw std::out_of_range("Attempt to access an inexistent measurement");
}

