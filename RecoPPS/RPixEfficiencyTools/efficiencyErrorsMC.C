// This function produces all the possible plane combinations extracting
// numberToExtract planes over numberOfPlanes planes
void getPlaneCombinations(
    const std::vector<uint32_t> &inputPlaneList, uint32_t numberToExtract,
    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>
        &planesExtractedAndNot) {
  uint32_t numberOfPlanes = inputPlaneList.size();
  std::string bitmask(numberToExtract, 1); // numberToExtract leading 1's
  bitmask.resize(numberOfPlanes,
                 0); // numberOfPlanes-numberToExtract trailing 0's
  planesExtractedAndNot.clear();

  // store the combination and permute bitmask
  do {
    planesExtractedAndNot.push_back(
        std::pair<std::vector<uint32_t>, std::vector<uint32_t>>(
            std::vector<uint32_t>(), std::vector<uint32_t>()));
    for (uint32_t i = 0; i < numberOfPlanes;
         ++i) { // [0..numberOfPlanes-1] integers
      if (bitmask[i])
        planesExtractedAndNot.back().second.push_back(inputPlaneList.at(i));
      else
        planesExtractedAndNot.back().first.push_back(inputPlaneList.at(i));
    }
  } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

  return;
}

float probabilityNplanesBlind(
    const std::vector<uint32_t> &inputPlaneList, int numberToExtract,
    const std::map<unsigned, float> &planeEfficiency) {
  std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>
      planesExtractedAndNot;
  getPlaneCombinations(inputPlaneList, numberToExtract, planesExtractedAndNot);

  float probability = 0.;

  for (const auto &combination : planesExtractedAndNot) {
    float combinationProbability = 1.;
    for (const auto &efficientPlane : combination.first) {
      combinationProbability *= planeEfficiency.at(efficientPlane);
    }
    for (const auto &notEfficientPlane : combination.second) {
      combinationProbability *= (1. - planeEfficiency.at(notEfficientPlane));
    }
    probability += combinationProbability;
  }
  return probability;
}

float probabilityCalculation(const std::map<uint32_t, float> &planeEfficiency) {

  std::vector<uint32_t> listOfPlanes_ = {0, 1, 2, 3, 4, 5};
  int minNumberOfBlindPlanes = 3;
  int maxNumberOfBlindPlanes = listOfPlanes_.size();
  float rpEfficiency = 1.;

  for (uint32_t i = (uint32_t)minNumberOfBlindPlanes;
       i <= (uint32_t)maxNumberOfBlindPlanes; i++) {
    rpEfficiency -= probabilityNplanesBlind(listOfPlanes_, i, planeEfficiency);
  }
  return rpEfficiency;
}

float simulatedEfficiency(const std::vector<uint32_t> &inputPlaneList,
                          const std::map<uint32_t, float> &planeTrueEfficiency,
                          const std::map<uint32_t, uint32_t> &eventsPerPlane) {
  std::map<uint32_t, float> planeEfficiency;
  for (const auto &plane : inputPlaneList) {
    planeEfficiency[plane] = gRandom->Binomial(eventsPerPlane.at(plane),
                                               planeTrueEfficiency.at(plane)) /
                             (float)eventsPerPlane.at(plane);
    // std::cout << "Plane " << plane << ": " << planeEfficiency[plane] << endl;                  
  }
  return probabilityCalculation(planeEfficiency);
}

TH1D simulation(int iterations, double eventsPerEfficiency = 50,
                double trueEfficiencyFlat = 0.98,
                double eventsPerPlaneFlat = 50) {

  TH1D hist = TH1D("Simulation", "Simulation;#varepsilon", 50, 0.99, 1);
  std::vector<uint32_t> planeList = {0, 1, 2, 3, 4, 5};
  std::map<uint32_t, float> planeTrueEfficiency;
  std::map<uint32_t, uint32_t> eventsPerPlane;

  for (auto &plane : planeList) {
    planeTrueEfficiency[plane] = trueEfficiencyFlat;
    eventsPerPlane[plane] = eventsPerPlaneFlat;
  }

  for (int it = 0; it < iterations; it++) {
    double effMeasurement = 0;
    if ((it + 1) % 10 == 0)
      std::cout << "Running iteration number " << it + 1 << "..." << std::endl;
    for (int i = 0; i < eventsPerEfficiency; i++) {
      effMeasurement +=
          simulatedEfficiency(planeList, planeTrueEfficiency, eventsPerPlane) /
          eventsPerEfficiency;
    }
    hist.Fill(effMeasurement);
    // std::cout << effMeasurement << std::endl;
  }
  return hist;
}

void efficiencyErrorsMC() {
  TH1D hist = simulation(500,5,0.98,5);
  hist.DrawCopy();
}