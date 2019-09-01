#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"
#include <iostream>
#include <cctype>  // for toupper
#include <algorithm>

EcalFedMap::~EcalFedMap() {}

EcalFedMap::EcalFedMap() {
  // EE-
  fedToSliceMap_.insert(std::make_pair(601, "EE-07"));
  fedToSliceMap_.insert(std::make_pair(602, "EE-08"));
  fedToSliceMap_.insert(std::make_pair(603, "EE-09"));
  fedToSliceMap_.insert(std::make_pair(604, "EE-01"));
  fedToSliceMap_.insert(std::make_pair(605, "EE-02"));
  fedToSliceMap_.insert(std::make_pair(606, "EE-03"));
  fedToSliceMap_.insert(std::make_pair(607, "EE-04"));
  fedToSliceMap_.insert(std::make_pair(608, "EE-05"));
  fedToSliceMap_.insert(std::make_pair(609, "EE-06"));

  // EB-
  fedToSliceMap_.insert(std::make_pair(610, "EB-01"));
  fedToSliceMap_.insert(std::make_pair(611, "EB-02"));
  fedToSliceMap_.insert(std::make_pair(612, "EB-03"));
  fedToSliceMap_.insert(std::make_pair(613, "EB-04"));
  fedToSliceMap_.insert(std::make_pair(614, "EB-05"));
  fedToSliceMap_.insert(std::make_pair(615, "EB-06"));
  fedToSliceMap_.insert(std::make_pair(616, "EB-07"));
  fedToSliceMap_.insert(std::make_pair(617, "EB-08"));
  fedToSliceMap_.insert(std::make_pair(618, "EB-09"));
  fedToSliceMap_.insert(std::make_pair(619, "EB-10"));
  fedToSliceMap_.insert(std::make_pair(620, "EB-11"));
  fedToSliceMap_.insert(std::make_pair(621, "EB-12"));
  fedToSliceMap_.insert(std::make_pair(622, "EB-13"));
  fedToSliceMap_.insert(std::make_pair(623, "EB-14"));
  fedToSliceMap_.insert(std::make_pair(624, "EB-15"));
  fedToSliceMap_.insert(std::make_pair(625, "EB-16"));
  fedToSliceMap_.insert(std::make_pair(626, "EB-17"));
  fedToSliceMap_.insert(std::make_pair(627, "EB-18"));

  // EB+
  fedToSliceMap_.insert(std::make_pair(628, "EB+01"));
  fedToSliceMap_.insert(std::make_pair(629, "EB+02"));
  fedToSliceMap_.insert(std::make_pair(630, "EB+03"));
  fedToSliceMap_.insert(std::make_pair(631, "EB+04"));
  fedToSliceMap_.insert(std::make_pair(632, "EB+05"));
  fedToSliceMap_.insert(std::make_pair(633, "EB+06"));
  fedToSliceMap_.insert(std::make_pair(634, "EB+07"));
  fedToSliceMap_.insert(std::make_pair(635, "EB+08"));
  fedToSliceMap_.insert(std::make_pair(636, "EB+09"));
  fedToSliceMap_.insert(std::make_pair(637, "EB+10"));
  fedToSliceMap_.insert(std::make_pair(638, "EB+11"));
  fedToSliceMap_.insert(std::make_pair(639, "EB+12"));
  fedToSliceMap_.insert(std::make_pair(640, "EB+13"));
  fedToSliceMap_.insert(std::make_pair(641, "EB+14"));
  fedToSliceMap_.insert(std::make_pair(642, "EB+15"));
  fedToSliceMap_.insert(std::make_pair(643, "EB+16"));
  fedToSliceMap_.insert(std::make_pair(644, "EB+17"));
  fedToSliceMap_.insert(std::make_pair(645, "EB+18"));

  // EE+
  fedToSliceMap_.insert(std::make_pair(646, "EE+07"));
  fedToSliceMap_.insert(std::make_pair(647, "EE+08"));
  fedToSliceMap_.insert(std::make_pair(648, "EE+09"));
  fedToSliceMap_.insert(std::make_pair(649, "EE+01"));
  fedToSliceMap_.insert(std::make_pair(650, "EE+02"));
  fedToSliceMap_.insert(std::make_pair(651, "EE+03"));
  fedToSliceMap_.insert(std::make_pair(652, "EE+04"));
  fedToSliceMap_.insert(std::make_pair(653, "EE+05"));
  fedToSliceMap_.insert(std::make_pair(654, "EE+06"));

  std::map<int, std::string>::iterator it;
  for (it = fedToSliceMap_.begin(); it != fedToSliceMap_.end(); it++) {
    //  std::cout<<  "fed: "<< (*it).first << " slice: " << (*it).second << std::endl;
    sliceToFedMap_.insert(std::make_pair((*it).second, (*it).first));
  }

  //  std::map<std::string, int>::iterator ti;
  //  for (ti = sliceToFedMap_.begin();
  //       ti != sliceToFedMap_.end();
  //       ti++)
  //    {
  //      //      std::cout<<  "slice: "<< (*ti).first << " fed: " << (*ti).second << std::endl;
  //    }
}

std::string EcalFedMap::getSliceFromFed(int fedNumber) {
  //std::cout << "received: " << fedNumber << std::endl;
  std::map<int, std::string>::iterator found = fedToSliceMap_.find(fedNumber);

  if (found != fedToSliceMap_.end())
    return (*found).second;
  else
    return std::string("invalid Fed");
}

int EcalFedMap::getFedFromSlice(std::string slice) {
  transform(slice.begin(), slice.end(), slice.begin(), toupper);

  std::map<std::string, int>::iterator found = sliceToFedMap_.find(slice);

  if (found != sliceToFedMap_.end())
    return (*found).second;
  else
    return -999;
}
