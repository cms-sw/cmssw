/*----------------------------------------------------------------------

Test of the statemachine classes.

----------------------------------------------------------------------*/

#include "FWCore/Framework/test/MockEventProcessor.h"

#include "catch2/catch_all.hpp"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

  using Transition = std::pair<char, int>;

  std::string toMockData(std::vector<Transition> const& transitions) {
    std::ostringstream data;
    bool first = true;
    for (auto const& transition : transitions) {
      if (not first) {
        data << '\n';
      }
      first = false;
      data << transition.first << ' ' << transition.second;
    }
    return data.str();
  }

  std::string extractModeOutput(std::string const& expectedCombinedOutput, bool mode) {
    std::string const nomergeHeader = "\nMachine parameters:  mode = NOMERGE\n";
    std::string const fullmergeHeader = "\nMachine parameters:  mode = FULLMERGE\n";

    auto const start = expectedCombinedOutput.find(mode ? nomergeHeader : fullmergeHeader);
    REQUIRE(start != std::string::npos);

    if (mode) {
      auto const end = expectedCombinedOutput.find(fullmergeHeader, start + nomergeHeader.size());
      REQUIRE(end != std::string::npos);
      return expectedCombinedOutput.substr(start, end - start);
    }
    return expectedCombinedOutput.substr(start);
  }

  std::string runStateMachine(std::vector<Transition> const& transitions, bool mode) {
    std::stringstream output;
    std::string const mockData = toMockData(transitions);

    output << "\nMachine parameters:  ";
    output << (mode ? "mode = NOMERGE" : "mode = FULLMERGE");
    output << "\n";

    edm::MockEventProcessor mockEventProcessor(mockData, output, mode);
    try {
      mockEventProcessor.runToCompletion();
    } catch (edm::MockEventProcessor::TestException const&) {
      output << "caught test exception\n";
    }
    return output.str();
  }

  void checkCase(std::string const& expectedCombined, std::vector<Transition> const& transitions) {
    REQUIRE_FALSE(expectedCombined.empty());
    SECTION("NOMERGE") {
      auto const expected = extractModeOutput(expectedCombined, true);
      auto const actual = runStateMachine(transitions, true);
      CHECK(actual == expected);
    }
    SECTION("FULLMERGE") {
      auto const expected = extractModeOutput(expectedCombined, false);
      auto const actual = runStateMachine(transitions, false);
      CHECK(actual == expected);
    }
  }

}  // namespace

TEST_CASE("MockEventProcessor statemachine", "[Framework]") {
  SECTION("case_1") {
    checkCase(R"SM(

Machine parameters:  mode = NOMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Run 2 ***
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	readRun 2
	beginRun 2
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 2/1
    *** nextItemType: Lumi 2 ***
	endLumi 2/1
	writeLumi 2/1
	clearLumiPrincipal 2/1
	readLuminosityBlock 2
	beginLumi 2/2
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Lumi 3 ***
	shouldWeStop
	endLumi 2/2
	writeLumi 2/2
	clearLumiPrincipal 2/2
	readLuminosityBlock 3
	beginLumi 2/3
    *** nextItemType: Run 3 ***
	endLumi 2/3
	writeLumi 2/3
	clearLumiPrincipal 2/3
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	readRun 3
	beginRun 3
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 3/1
    *** nextItemType: Stop 1 ***
	endLumi 3/1
	writeLumi 3/1
	clearLumiPrincipal 3/1
	endRun 3
	writeRun 3
	clearRunPrincipal 3
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.

Machine parameters:  mode = FULLMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Run 2 ***
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	readRun 2
	beginRun 2
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 2/1
    *** nextItemType: Lumi 2 ***
	endLumi 2/1
	writeLumi 2/1
	clearLumiPrincipal 2/1
	readLuminosityBlock 2
	beginLumi 2/2
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Lumi 3 ***
	shouldWeStop
	endLumi 2/2
	writeLumi 2/2
	clearLumiPrincipal 2/2
	readLuminosityBlock 3
	beginLumi 2/3
    *** nextItemType: Run 3 ***
	endLumi 2/3
	writeLumi 2/3
	clearLumiPrincipal 2/3
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	readRun 3
	beginRun 3
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 3/1
    *** nextItemType: Stop 1 ***
	endLumi 3/1
	writeLumi 3/1
	clearLumiPrincipal 3/1
	endRun 3
	writeRun 3
	clearRunPrincipal 3
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.
)SM",
              {{'f', 1}, {'r', 1}, {'r', 2}, {'l', 1}, {'l', 2}, {'e', 1}, {'l', 3}, {'r', 3}, {'l', 1}, {'s', 1}});
  }
  SECTION("case_2") {
    checkCase(R"SM(

Machine parameters:  mode = NOMERGE
	startingNewLoop
    *** nextItemType: File 0 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: File 1 ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: File 1 ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: File 0 ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: File 0 ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: File 1 ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: File 0 ***
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: File 0 ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: File 1 ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 2 ***
	readRun 2
	beginRun 2
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 2/1
    *** nextItemType: File 0 ***
	endLumi 2/1
	writeLumi 2/1
	clearLumiPrincipal 2/1
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: File 0 ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: File 1 ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Stop 1 ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.

Machine parameters:  mode = FULLMERGE
	startingNewLoop
    *** nextItemType: File 0 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: File 1 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: File 1 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: File 0 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: File 0 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: File 1 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: File 0 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: File 0 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: File 1 ***
	shouldWeCloseOutput
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 2 ***
	readRun 2
	beginRun 2
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 2/1
    *** nextItemType: File 0 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: File 0 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: File 1 ***
	shouldWeCloseOutput
	endLumi 2/1
	writeLumi 2/1
	clearLumiPrincipal 2/1
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Stop 1 ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.
)SM",
              {{'f', 0},
               {'f', 1},
               {'f', 1},
               {'f', 0},
               {'f', 0},
               {'f', 1},
               {'r', 1},
               {'f', 0},
               {'f', 0},
               {'f', 1},
               {'r', 2},
               {'l', 1},
               {'f', 0},
               {'f', 0},
               {'f', 1},
               {'s', 1}});
  }
  SECTION("case_3") {
    checkCase(R"SM(

Machine parameters:  mode = NOMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 2 ***
	readRun 2
	beginRun 2
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 2/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	endLumi 2/1
	writeLumi 2/1
	clearLumiPrincipal 2/1
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 2 ***
	readLuminosityBlock 2
	beginLumi 1/2
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Stop 1 ***
	shouldWeStop
	endLumi 1/2
	writeLumi 1/2
	clearLumiPrincipal 1/2
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.

Machine parameters:  mode = FULLMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 2 ***
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	readRun 2
	beginRun 2
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 2/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 1 ***
	endLumi 2/1
	writeLumi 2/1
	clearLumiPrincipal 2/1
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 2 ***
	readLuminosityBlock 2
	beginLumi 1/2
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Stop 1 ***
	shouldWeStop
	endLumi 1/2
	writeLumi 1/2
	clearLumiPrincipal 1/2
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.
)SM",
              {{'f', 1},
               {'r', 1},
               {'l', 1},
               {'e', 1},
               {'f', 0},
               {'r', 2},
               {'l', 1},
               {'e', 1},
               {'f', 0},
               {'r', 1},
               {'l', 2},
               {'e', 2},
               {'s', 1}});
  }
  SECTION("case_4") {
    checkCase(R"SM(

Machine parameters:  mode = NOMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Run 2 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	readRun 2
	beginRun 2
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 2/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	endLumi 2/1
	writeLumi 2/1
	clearLumiPrincipal 2/1
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 2 ***
	readLuminosityBlock 2
	beginLumi 1/2
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Stop 1 ***
	shouldWeStop
	endLumi 1/2
	writeLumi 1/2
	clearLumiPrincipal 1/2
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.

Machine parameters:  mode = FULLMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Run 2 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	readRun 2
	beginRun 2
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 2/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 1 ***
	endLumi 2/1
	writeLumi 2/1
	clearLumiPrincipal 2/1
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 2 ***
	readLuminosityBlock 2
	beginLumi 1/2
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Stop 1 ***
	shouldWeStop
	endLumi 1/2
	writeLumi 1/2
	clearLumiPrincipal 1/2
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.
)SM",
              {{'f', 1},
               {'r', 1},
               {'l', 1},
               {'e', 1},
               {'r', 2},
               {'l', 1},
               {'e', 1},
               {'f', 0},
               {'r', 1},
               {'l', 2},
               {'e', 2},
               {'s', 1}});
  }
  SECTION("case_5") {
    checkCase(R"SM(

Machine parameters:  mode = NOMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Lumi 2 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	readLuminosityBlock 2
	beginLumi 1/2
    *** nextItemType: File 0 ***
	endLumi 1/2
	writeLumi 1/2
	clearLumiPrincipal 1/2
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 2 ***
	readLuminosityBlock 2
	beginLumi 1/2
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Stop 1 ***
	shouldWeStop
	endLumi 1/2
	writeLumi 1/2
	clearLumiPrincipal 1/2
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.

Machine parameters:  mode = FULLMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Lumi 2 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	readLuminosityBlock 2
	beginLumi 1/2
    *** nextItemType: File 0 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 1 ***
	readAndMergeRun 1
    *** nextItemType: Lumi 2 ***
	readAndMergeLumi 2
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Stop 1 ***
	shouldWeStop
	endLumi 1/2
	writeLumi 1/2
	clearLumiPrincipal 1/2
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.
)SM",
              {{'f', 1}, {'r', 1}, {'l', 1}, {'e', 1}, {'l', 2}, {'f', 0}, {'r', 1}, {'l', 2}, {'e', 1}, {'s', 1}});
  }
  SECTION("case_6") {
    checkCase(
        R"SM(

Machine parameters:  mode = NOMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Lumi 2 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	readLuminosityBlock 2
	beginLumi 1/2
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	endLumi 1/2
	writeLumi 1/2
	clearLumiPrincipal 1/2
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 2 ***
	readLuminosityBlock 2
	beginLumi 1/2
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Stop 1 ***
	shouldWeStop
	endLumi 1/2
	writeLumi 1/2
	clearLumiPrincipal 1/2
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.

Machine parameters:  mode = FULLMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Lumi 2 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	readLuminosityBlock 2
	beginLumi 1/2
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 1 ***
	readAndMergeRun 1
    *** nextItemType: Lumi 2 ***
	readAndMergeLumi 2
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Stop 1 ***
	shouldWeStop
	endLumi 1/2
	writeLumi 1/2
	clearLumiPrincipal 1/2
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.
)SM",
        {{'f', 1}, {'r', 1}, {'l', 1}, {'e', 1}, {'l', 2}, {'e', 1}, {'f', 0}, {'r', 1}, {'l', 2}, {'e', 2}, {'s', 1}});
  }
  SECTION("case_7") {
    checkCase(R"SM(

Machine parameters:  mode = NOMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Lumi 2 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	readLuminosityBlock 2
	beginLumi 1/2
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	endLumi 1/2
	writeLumi 1/2
	clearLumiPrincipal 1/2
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 2 ***
	readLuminosityBlock 2
	beginLumi 1/2
    *** nextItemType: Lumi 3 ***
	endLumi 1/2
	writeLumi 1/2
	clearLumiPrincipal 1/2
	readLuminosityBlock 3
	beginLumi 1/3
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Stop 1 ***
	shouldWeStop
	endLumi 1/3
	writeLumi 1/3
	clearLumiPrincipal 1/3
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.

Machine parameters:  mode = FULLMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Lumi 2 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	readLuminosityBlock 2
	beginLumi 1/2
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 1 ***
	readAndMergeRun 1
    *** nextItemType: Lumi 2 ***
	readAndMergeLumi 2
    *** nextItemType: Lumi 3 ***
	endLumi 1/2
	writeLumi 1/2
	clearLumiPrincipal 1/2
	readLuminosityBlock 3
	beginLumi 1/3
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Stop 1 ***
	shouldWeStop
	endLumi 1/3
	writeLumi 1/3
	clearLumiPrincipal 1/3
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.
)SM",
              {{'f', 1},
               {'r', 1},
               {'l', 1},
               {'e', 1},
               {'l', 2},
               {'e', 1},
               {'f', 0},
               {'r', 1},
               {'l', 2},
               {'l', 3},
               {'e', 1},
               {'s', 1}});
  }
  SECTION("case_8") {
    checkCase(
        R"SM(

Machine parameters:  mode = NOMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Lumi 2 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	readLuminosityBlock 2
	beginLumi 1/2
    *** nextItemType: File 0 ***
	endLumi 1/2
	writeLumi 1/2
	clearLumiPrincipal 1/2
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 2 ***
	readLuminosityBlock 2
	beginLumi 1/2
    *** nextItemType: Lumi 3 ***
	endLumi 1/2
	writeLumi 1/2
	clearLumiPrincipal 1/2
	readLuminosityBlock 3
	beginLumi 1/3
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Stop 1 ***
	shouldWeStop
	endLumi 1/3
	writeLumi 1/3
	clearLumiPrincipal 1/3
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.

Machine parameters:  mode = FULLMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Lumi 2 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	readLuminosityBlock 2
	beginLumi 1/2
    *** nextItemType: File 0 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 1 ***
	readAndMergeRun 1
    *** nextItemType: Lumi 2 ***
	readAndMergeLumi 2
    *** nextItemType: Lumi 3 ***
	endLumi 1/2
	writeLumi 1/2
	clearLumiPrincipal 1/2
	readLuminosityBlock 3
	beginLumi 1/3
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Stop 1 ***
	shouldWeStop
	endLumi 1/3
	writeLumi 1/3
	clearLumiPrincipal 1/3
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.
)SM",
        {{'f', 1}, {'r', 1}, {'l', 1}, {'e', 1}, {'l', 2}, {'f', 0}, {'r', 1}, {'l', 2}, {'l', 3}, {'e', 1}, {'s', 1}});
  }
  SECTION("case_9") {
    checkCase(R"SM(

Machine parameters:  mode = NOMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Event ***
	shouldWeStop
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Event ***
	shouldWeStop
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Event ***
	shouldWeStop
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Event ***
    *** shouldWeStop will return true this event ***
	shouldWeStop
	readEvent
	processEvent
	shouldWeStop
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.
	startingNewLoop
    *** nextItemType: Event ***
	doErrorStuff
Left processing loop.
	startingNewLoop
    *** nextItemType: Stop 1 ***
	endOfLoop
Left processing loop.

Machine parameters:  mode = FULLMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Event ***
	shouldWeStop
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Event ***
	shouldWeStop
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 1 ***
	readAndMergeRun 1
    *** nextItemType: Lumi 1 ***
	readAndMergeLumi 1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Event ***
	shouldWeStop
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 1 ***
	readAndMergeRun 1
    *** nextItemType: Lumi 1 ***
	readAndMergeLumi 1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Event ***
    *** shouldWeStop will return true this event ***
	shouldWeStop
	readEvent
	processEvent
	shouldWeStop
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.
	startingNewLoop
    *** nextItemType: Event ***
	doErrorStuff
Left processing loop.
	startingNewLoop
    *** nextItemType: Stop 1 ***
	endOfLoop
Left processing loop.
)SM",
              {{'f', 1},
               {'r', 1},
               {'l', 1},
               {'e', 1},
               {'e', 2},
               {'e', 3},
               {'f', 0},
               {'r', 1},
               {'l', 1},
               {'e', 4},
               {'e', 5},
               {'f', 0},
               {'r', 1},
               {'l', 1},
               {'e', 6},
               {'e', 7},
               {'e', 8},
               {'s', 1}});
  }
  SECTION("case_10") {
    checkCase(R"SM(

Machine parameters:  mode = NOMERGE
	startingNewLoop
    *** nextItemType: Restart 0 ***
	endOfLoop
	prepareForNextLoop
	rewind
	startingNewLoop
    *** nextItemType: Event ***
	doErrorStuff
Left processing loop.
	startingNewLoop
    *** nextItemType: Lumi 1 ***
	doErrorStuff
Left processing loop.
	startingNewLoop
    *** nextItemType: Run 1 ***
	doErrorStuff
Left processing loop.
	startingNewLoop
    *** nextItemType: Stop 1 ***
	endOfLoop
Left processing loop.

Machine parameters:  mode = FULLMERGE
	startingNewLoop
    *** nextItemType: Restart 0 ***
	endOfLoop
	prepareForNextLoop
	rewind
	startingNewLoop
    *** nextItemType: Event ***
	doErrorStuff
Left processing loop.
	startingNewLoop
    *** nextItemType: Lumi 1 ***
	doErrorStuff
Left processing loop.
	startingNewLoop
    *** nextItemType: Run 1 ***
	doErrorStuff
Left processing loop.
	startingNewLoop
    *** nextItemType: Stop 1 ***
	endOfLoop
Left processing loop.
)SM",
              {{'x', 0}, {'e', 1}, {'l', 1}, {'r', 1}, {'s', 1}});
  }
  SECTION("case_11") {
    checkCase(
        R"SM(

Machine parameters:  mode = NOMERGE
	startingNewLoop
    *** nextItemType: Restart 0 ***
	endOfLoop
	prepareForNextLoop
	rewind
	startingNewLoop
    *** nextItemType: File 0 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Restart 0 ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
	prepareForNextLoop
	rewind
	startingNewLoop
    *** nextItemType: File 0 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Lumi 1 ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	doErrorStuff
Left processing loop.
	startingNewLoop
    *** nextItemType: File 0 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Event ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	doErrorStuff
Left processing loop.
	startingNewLoop
    *** nextItemType: File 0 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: File 0 ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 9 ***
	readRun 9
	beginRun 9
    *** nextItemType: File 0 ***
	endRun 9
	writeRun 9
	clearRunPrincipal 9
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: File 0 ***
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Run 2 ***
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	readRun 2
	beginRun 2
    *** nextItemType: File 0 ***
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 2 ***
	readRun 2
	beginRun 2
    *** nextItemType: File 0 ***
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 2 ***
	readRun 2
	beginRun 2
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 2/1
    *** nextItemType: File 0 ***
	endLumi 2/1
	writeLumi 2/1
	clearLumiPrincipal 2/1
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 2 ***
	readRun 2
	beginRun 2
    *** nextItemType: Lumi 2 ***
	readLuminosityBlock 2
	beginLumi 2/2
    *** nextItemType: Lumi 3 ***
	endLumi 2/2
	writeLumi 2/2
	clearLumiPrincipal 2/2
	readLuminosityBlock 3
	beginLumi 2/3
    *** nextItemType: File 0 ***
	endLumi 2/3
	writeLumi 2/3
	clearLumiPrincipal 2/3
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 2 ***
	readRun 2
	beginRun 2
    *** nextItemType: File 0 ***
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 2 ***
	readRun 2
	beginRun 2
    *** nextItemType: Lumi 3 ***
	readLuminosityBlock 3
	beginLumi 2/3
    *** nextItemType: File 0 ***
	endLumi 2/3
	writeLumi 2/3
	clearLumiPrincipal 2/3
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Lumi 4 ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	doErrorStuff
Left processing loop.
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 5 ***
	readRun 5
	beginRun 5
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 5/1
    *** nextItemType: File 0 ***
	endLumi 5/1
	writeLumi 5/1
	clearLumiPrincipal 5/1
	endRun 5
	writeRun 5
	clearRunPrincipal 5
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Event ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	doErrorStuff
Left processing loop.
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 6 ***
	readRun 6
	beginRun 6
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 6/1
    *** nextItemType: Restart 0 ***
	endLumi 6/1
	writeLumi 6/1
	clearLumiPrincipal 6/1
	endRun 6
	writeRun 6
	clearRunPrincipal 6
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
	prepareForNextLoop
	rewind
	startingNewLoop
    *** nextItemType: Stop 1 ***
	endOfLoop
Left processing loop.

Machine parameters:  mode = FULLMERGE
	startingNewLoop
    *** nextItemType: Restart 0 ***
	endOfLoop
	prepareForNextLoop
	rewind
	startingNewLoop
    *** nextItemType: File 0 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Restart 0 ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
	prepareForNextLoop
	rewind
	startingNewLoop
    *** nextItemType: File 0 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Lumi 1 ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	doErrorStuff
Left processing loop.
	startingNewLoop
    *** nextItemType: File 0 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Event ***
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	doErrorStuff
Left processing loop.
	startingNewLoop
    *** nextItemType: File 0 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: File 0 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 9 ***
	readRun 9
	beginRun 9
    *** nextItemType: File 0 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 1 ***
	endRun 9
	writeRun 9
	clearRunPrincipal 9
	readRun 1
	beginRun 1
    *** nextItemType: File 0 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 1 ***
	readAndMergeRun 1
    *** nextItemType: Run 2 ***
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	readRun 2
	beginRun 2
    *** nextItemType: File 0 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 2 ***
	readAndMergeRun 2
    *** nextItemType: File 0 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 2 ***
	readAndMergeRun 2
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 2/1
    *** nextItemType: File 0 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 2 ***
	readAndMergeRun 2
    *** nextItemType: Lumi 2 ***
	endLumi 2/1
	writeLumi 2/1
	clearLumiPrincipal 2/1
	readLuminosityBlock 2
	beginLumi 2/2
    *** nextItemType: Lumi 3 ***
	endLumi 2/2
	writeLumi 2/2
	clearLumiPrincipal 2/2
	readLuminosityBlock 3
	beginLumi 2/3
    *** nextItemType: File 0 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 2 ***
	readAndMergeRun 2
    *** nextItemType: File 0 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 2 ***
	readAndMergeRun 2
    *** nextItemType: Lumi 3 ***
	readAndMergeLumi 3
    *** nextItemType: File 0 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Lumi 4 ***
	endLumi 2/3
	writeLumi 2/3
	clearLumiPrincipal 2/3
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	doErrorStuff
Left processing loop.
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 5 ***
	readRun 5
	beginRun 5
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 5/1
    *** nextItemType: File 0 ***
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Event ***
	endLumi 5/1
	writeLumi 5/1
	clearLumiPrincipal 5/1
	endRun 5
	writeRun 5
	clearRunPrincipal 5
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	doErrorStuff
Left processing loop.
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 6 ***
	readRun 6
	beginRun 6
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 6/1
    *** nextItemType: Restart 0 ***
	endLumi 6/1
	writeLumi 6/1
	clearLumiPrincipal 6/1
	endRun 6
	writeRun 6
	clearRunPrincipal 6
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
	prepareForNextLoop
	rewind
	startingNewLoop
    *** nextItemType: Stop 1 ***
	endOfLoop
Left processing loop.
)SM",
        {{'x', 0}, {'f', 0}, {'x', 0}, {'f', 0}, {'l', 1}, {'f', 0}, {'e', 1}, {'f', 0}, {'f', 0}, {'r', 9}, {'f', 0},
         {'r', 1}, {'f', 0}, {'r', 1}, {'r', 2}, {'f', 0}, {'r', 2}, {'f', 0}, {'r', 2}, {'l', 1}, {'f', 0}, {'r', 2},
         {'l', 2}, {'l', 3}, {'f', 0}, {'r', 2}, {'f', 0}, {'r', 2}, {'l', 3}, {'f', 0}, {'l', 4}, {'f', 1}, {'r', 5},
         {'l', 1}, {'f', 0}, {'e', 1}, {'f', 1}, {'r', 6}, {'l', 1}, {'x', 0}, {'s', 1}});
  }
  SECTION("case_12") {
    checkCase(R"SM(

Machine parameters:  mode = NOMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 2 ***
	readLuminosityBlock 2
	beginLumi 1/2
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	endLumi 1/2
	writeLumi 1/2
	clearLumiPrincipal 1/2
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Stop 1 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.

Machine parameters:  mode = FULLMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 1 ***
	readAndMergeRun 1
    *** nextItemType: Lumi 2 ***
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	readLuminosityBlock 2
	beginLumi 1/2
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 1 ***
	readAndMergeRun 1
    *** nextItemType: Lumi 1 ***
	endLumi 1/2
	writeLumi 1/2
	clearLumiPrincipal 1/2
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Stop 1 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
	endOfLoop
Left processing loop.
)SM",
              {{'f', 1},
               {'r', 1},
               {'l', 1},
               {'e', 1},
               {'f', 0},
               {'r', 1},
               {'l', 2},
               {'e', 1},
               {'f', 0},
               {'r', 1},
               {'l', 1},
               {'e', 2},
               {'s', 1}});
  }
  SECTION("case_20") {
    checkCase(R"SM(

Machine parameters:  mode = NOMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 2 ***
	readRun 2
	beginRun 2
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 2/1
    *** nextItemType: Throw 1 ***
    *** nextItemType: Event ***
	readEvent
	processEvent
	throwing
	endLumi 2/1
	writeLumi 2/1
	clearLumiPrincipal 2/1
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
caught test exception

Machine parameters:  mode = FULLMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 2 ***
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	readRun 2
	beginRun 2
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 2/1
    *** nextItemType: Throw 1 ***
    *** nextItemType: Event ***
	readEvent
	processEvent
	throwing
	endLumi 2/1
	writeLumi 2/1
	clearLumiPrincipal 2/1
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
caught test exception
)SM",
              {{'f', 1}, {'r', 1}, {'l', 1}, {'e', 1}, {'f', 0}, {'r', 2}, {'l', 1}, {'t', 1}, {'e', 1}});
  }
  SECTION("case_21") {
    checkCase(R"SM(

Machine parameters:  mode = NOMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 2 ***
	readRun 2
	beginRun 2
    *** nextItemType: Throw 1 ***
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 2/1
	throwing
	endLumi 2/1 global failed
	clearLumiPrincipal 2/1
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
caught test exception

Machine parameters:  mode = FULLMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Run 2 ***
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	readRun 2
	beginRun 2
    *** nextItemType: Throw 1 ***
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 2/1
	throwing
	endLumi 2/1 global failed
	clearLumiPrincipal 2/1
	endRun 2
	writeRun 2
	clearRunPrincipal 2
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
caught test exception
)SM",
              {{'f', 1}, {'r', 1}, {'l', 1}, {'e', 1}, {'f', 0}, {'r', 2}, {'t', 1}, {'l', 1}});
  }
  SECTION("case_22") {
    checkCase(R"SM(

Machine parameters:  mode = NOMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Throw 1 ***
    *** nextItemType: Run 2 ***
	readRun 2
	beginRun 2
	throwing
	endRun 2 global failed
	clearRunPrincipal 2
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
caught test exception

Machine parameters:  mode = FULLMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: File 0 ***
	shouldWeStop
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	respondToOpenInputFile
    *** nextItemType: Throw 1 ***
    *** nextItemType: Run 2 ***
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	readRun 2
	beginRun 2
	throwing
	endRun 2 global failed
	clearRunPrincipal 2
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
caught test exception
)SM",
              {{'f', 1}, {'r', 1}, {'l', 1}, {'e', 1}, {'f', 0}, {'t', 1}, {'r', 2}});
  }
  SECTION("case_23") {
    checkCase(R"SM(

Machine parameters:  mode = NOMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Throw 1 ***
    *** nextItemType: File 0 ***
	shouldWeStop
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
 	readFile
	throwing
	respondToCloseInputFile
	closeInputFile
caught test exception

Machine parameters:  mode = FULLMERGE
	startingNewLoop
    *** nextItemType: File 1 ***
 	readFile
	respondToOpenInputFile
	openOutputFiles
    *** nextItemType: Run 1 ***
	readRun 1
	beginRun 1
    *** nextItemType: Lumi 1 ***
	readLuminosityBlock 1
	beginLumi 1/1
    *** nextItemType: Event ***
	readEvent
	processEvent
	shouldWeStop
    *** nextItemType: Throw 1 ***
    *** nextItemType: File 0 ***
	shouldWeStop
	shouldWeCloseOutput
	respondToCloseInputFile
	closeInputFile
 	readFile
	throwing
	endLumi 1/1
	writeLumi 1/1
	clearLumiPrincipal 1/1
	endRun 1
	writeRun 1
	clearRunPrincipal 1
	respondToCloseInputFile
	closeInputFile
	closeOutputFiles
caught test exception
)SM",
              {{'f', 1}, {'r', 1}, {'l', 1}, {'e', 1}, {'t', 1}, {'f', 0}});
  }
}
