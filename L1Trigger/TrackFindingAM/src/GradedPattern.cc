#include "../interface/GradedPattern.h"

GradedPattern::GradedPattern():Pattern(){
  grade=0;
  averagePt=0;
}

GradedPattern::GradedPattern(const Pattern& p):Pattern(p){
  grade=0;
  averagePt=0;
}

int GradedPattern::getGrade() const{
  return grade;
}

float GradedPattern::getAveragePt() const{
  return averagePt;
}

void GradedPattern::increment(){
  grade++;
}

void GradedPattern::increment(float pt){
  averagePt=(averagePt*grade+pt)/(grade+1);
  grade++;
}

int GradedPattern::operator<(const GradedPattern& gp){
  return grade<gp.getGrade();
}
