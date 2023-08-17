//initializing the libraries needed
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <vector>
#include <random>
#include <utility>
#include <mpi.h>
#include <unistd.h>
#include "workerMPI_process.h"
#include "masterMPI_process.h"
#include "processInterface.h"
#include <cassert>
#include <tuple>

//Implement MPI_sendrecv

const int MPI_METHODS_COUNT = 7;

std::tuple<int, std::vector<int>, int, int> parseCommands(int argc, char* argv[]) {
  enum INPUT_OPTIONS { VECTOR_SIZE = 's', COMMUNICATION_METHOD = 'f', ITERATIONS = 'i' };

  //default values
  int vecSize = 100;
  int iterations = 10;
  std::vector<int> commMethods;

  int input;      // Parsing command-line arguments
  int inputNum_;  // Parsing command-line arguments

  while ((input = getopt(argc, argv, "s:f:i:")) != -1) {
    switch (input) {
      case VECTOR_SIZE:
        try {
          vecSize = std::stoll(optarg, nullptr, 0);
        } catch (std::exception& err) {
          std::cout << "\n\tError: Argument s must be an integer!";
          std::cout << "\n\t" << err.what() << std::endl;
          abort();
        }
        break;
      case COMMUNICATION_METHOD:
        try {
          // Sending Methods selected by user (e.g. 34 user selected methods blocking and nonblocking)
          inputNum_ = std::stoll(optarg, nullptr, 0);
          int inputNum = inputNum_;
          while (inputNum > 0) {
            int digit = inputNum % 10;
            if (digit > MPI_METHODS_COUNT) {
              //FIXME: Raise an exception here
              std::cout << "\n\tError: Argument must be an integer <= " << MPI_METHODS_COUNT << std::endl;
              abort();
            }
            commMethods.push_back(digit);
            inputNum = inputNum / 10;
          }
        } catch (std::exception& err) {
          std::cout << "\n\tError: Argument r must be an integer!";
          std::cout << "\n\t" << err.what() << std::endl;
          abort();
        }
        break;
      case ITERATIONS:
        try {
          // Set the average run count based on the command-line argument.
          iterations = std::stoll(optarg, nullptr, 0);
        } catch (std::exception& err) {
          std::cout << "\n\tError: Argument n must be an integer!";
          std::cout << "\n\t" << err.what() << std::endl;
          abort();
        }
        break;
      default:
        std::cerr << "\n\t WRONGE INPUT ****** ABORT! " << input << "\n";
        abort();
    }
  }
  return std::make_tuple(vecSize, commMethods, iterations, inputNum_);
}

// mian function
int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank;
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  assert(size >= 2);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto [vecSize, commMethods, iterations, inputNum_] = parseCommands(argc, argv);

  std::unique_ptr<MPI_TEST> MPIObject;

  if (rank == 0) {
    MPIObject = std::make_unique<MPI_Master>(vecSize);
  } else {
    MPIObject = std::make_unique<MPI_Worker>();
  }

  MPIObject->getTimeMeasurements(vecSize, commMethods, iterations, inputNum_);

  MPI_Finalize();

  return 0;
}
