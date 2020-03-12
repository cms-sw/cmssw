#include "Alignment/CommonAlignment/interface/Alignable.h"

#include "Alignment/SurveyAnalysis/interface/SurveyInputBase.h"

Alignable* SurveyInputBase::theDetector(nullptr);

SurveyInputBase::~SurveyInputBase() {
  delete theDetector;

  theDetector = nullptr;
}

void SurveyInputBase::addComponent(Alignable* comp) {
  if (nullptr == theDetector)
    theDetector = comp;
  else
    theDetector->addComponent(comp);
}
