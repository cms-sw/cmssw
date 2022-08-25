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
//////////////////////////////////////////// C U D A  /////////////////////////////////////////
#include <cuda.h>
#include <thrust/device_vector.h>
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

//called in the Host and excuted in the Device (GPU)
__global__ void addVectorsGpu(float *vect1, float *vect2, float *vect3, int size, int taskN) {
  //blockDim.x gives the number of threads in a block, in the x direction.
  //gridDim.x gives the number of blocks in a grid, in the x direction.
  //blockDim.x * gridDim.x gives the number of threads in a grid (in the x direction, in this case).
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = 0; i < taskN; ++i) {
    for (int j = first; j < size; j += stride) {
      vect3[j] = vect2[j] + vect1[j];
    }
  }
}  //add two vectors and save the result into the third vector.
//////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////  Global Varaibles  /////////////////////////////////////
unsigned int sizeVector = 2000;
int average = 5;
int task = 1;
int partsToRun = 1;
bool printStander = false;
bool saveFile = false;
bool help = false;
//////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////  Data Structure  /////////////////////////////////////
//Data For User's Choises Only
struct UserChoises {
  // unsigned int sizeVector;
  unsigned int sizeVectorBytes;  //Defualt vectors element float
  // unsigned int average;
  unsigned int extra;
  // unsigned int task;
  // unsigned int partsToRun;
  int root;
  // int numberProcess;
  int averageVectorSend;
  std::vector<int> partsToRunVector;  //vector for user's choice of part.
};

//Data For MPI Only
struct MPIData {
  int num_procs;
  int rank;

  std::pair<int, int> workSplit;
  float *mVect1;                  //declare vector 1.
  float *mVect2;                  //declare vector 2.
  float *mVect3;                  //declare vector fulled only by root to get result from workers.
  float *mVectChecking;           //declare vector to verify the results from each process.
  float *mVectWorker1;            //declare vector 1 for workers only.
  float *mVectWorker2;            //declare vector 2 for workers only.
  float *mVectWorker3;            //declare vector 2 for workers only.
  std::vector<int> displacement;  //declare vector for selecting location of each element to be sent.
  std::vector<int> numberToSend;
};

//Data For Cuda Only
struct Pointers {
  float *vect1;  //pointers only for Host
  float *vect2;
  float *vect3;

  float *dVect1;  //pointers only for device
  float *dVect2;
  float *dVect3;

  float *dVect1Extra;  //pointers only for device
  float *dVect2Extra;
  float *dVect3Extra;
};

//Data for Time Measurements Only
struct Timing {
  int partChosen;
  int unitChoice;
  double inputPreparationRoot[2];     // get time points from start and end on Root Side
  double inputPreparationHost[2];     // get time points from start and end on Host Side.
  double operationOnDeviceByHost[2];  //get time duration in Device with Host perspective.

  double outputPreparationRoot[2];
  double outputPreparationHost[2];

  std::vector<double> timeInputPreparationRoot;  //Save the Duration time.
  std::vector<double> timeInputPreparationHost;
  std::vector<double> timeOperationOnDeviceByRootHost;
  std::vector<double> timeOutputPreparationRoot;
  std::vector<double> timeOutputPreparationHost;

  cudaEvent_t start, stop;                          //get time points in Device.
  float operationOnDeviceByDevice = 0;              //get time duration in Device with device perspective.
  std::vector<float> operationOnDeviceByDeviceAcc;  //get accumulating time duration in Device with device perspective.
  std::vector<float> averageResults;  ///declare vector for getting average calcualtion for Hosts and device to Root.
};

//////////////////////////////////////////////////////////////////////////////////////////////////
const std::vector<int> chooseFunction(int toInteger);
std::pair<int, int> splitProcess(int works, int numberOfProcess);
const std::vector<int> numberDataSend(int numberOfProcess, std::pair<int, int> splitWorks);
void setupMPIAndVectors(
    MPIData &mpiData,
    UserChoises &user);  //initialize communicator environment for MPI and Resize Vectors with Generating Random numbers.
void setupTime(Timing &timing, UserChoises &user);  //Resizing Vectors of Time.
void calculateTimeDuration(Timing &timing, int i, int &root);
void addVectorsHost(float *vect1, float *vect2, float *vect3);
void cleanBuffer(float *vect);
bool checkingResultsPrintout(float *vectCpu, float *vectGpu);
void calculateAverageDeviation(Timing &timing, int averg, int &root);
bool sendAverageToRoot(Timing &timing, UserChoises &user, int &rank);

Timing blockSendPart1(MPIData &mpidata, Timing &timing, Pointers &pointer, UserChoises &user);
Timing blockSendPart2(MPIData &mpiData, Timing &timing, Pointers &pointer, UserChoises &user);
Timing blockSendPart3(MPIData &mpiData, Timing &timing, Pointers &pointer, UserChoises &user);

void printTable(std::vector<Timing> &timing, bool standerDeviationPrint);
int getNumberofDigits(double number);
void newLineTitle(int line, const std::string &title);
void printResultEach(std::vector<Timing> &timing, int type, bool standerDeviationPrint);
bool saveToFile(const std::string &name, const Timing &timing);

void printHelp(void);
int main(int argc, char *argv[]) {
  int c;  //to get parameters from user.

  UserChoises user;  //Setup Uuser's input variables
  user.extra = 2;
  user.root = 0;
  user.averageVectorSend = 8;

  while ((c = getopt(argc, argv, "s:a:t:p:qfh")) != -1) {
    switch (c) {
      case 's':
        try {
          sizeVector = std::stoll(optarg, nullptr, 0);
        } catch (std::exception &err) {
          std::cout << "\n\tError Must be integer Argument!";
          std::cout << "\n\t" << err.what() << std::endl;
          return 0;
        }
        break;
      case 'a':
        try {
          average = std::stoll(optarg, nullptr, 0);

        } catch (std::exception &err) {
          std::cout << "\n\tError Must be integer Argument!";
          std::cout << "\n\t" << err.what() << std::endl;
          return 0;
        }
        break;
      case 't':
        try {
          task = std::stoll(optarg, nullptr, 0);
          //std::cout << "\nNumber of repeated Task is " << task << std::endl;
        } catch (std::exception &err) {
          std::cout << "\n\tError Must be integer Argument!";
          std::cout << "\n\t" << err.what() << std::endl;
          return 0;
        }
        break;
      case 'p':
        try {
          partsToRun = std::stoll(optarg, nullptr, 0);
          user.partsToRunVector = chooseFunction(partsToRun);
          //std::cout << "\nyou have chosen Part ";
          for (unsigned int j = 0; j < user.partsToRunVector.size(); ++j) {
            std::cout << user.partsToRunVector[j] << " ,";
          }
          std::cout << "\n";
        } catch (std::exception &err) {
          std::cout << "\n\tError Must be integer Argument!";
          std::cout << "\n\t" << err.what() << std::endl;
          return 0;
        }
        break;
      case 'q':
        try {
          printStander = true;
        } catch (std::exception &err) {
          std::cout << "\n\tError Must be integer Argument!";
          std::cout << "\n\t" << err.what() << std::endl;
          return 0;
        }
        break;
      case 'f':
        try {
          saveFile = true;
        } catch (std::exception &err) {
          std::cout << "\n\tError Must be integer Argument!";
          std::cout << "\n\t" << err.what() << std::endl;
          return 0;
        }
        break;
      case 'h':
        try {
          help = true;
        } catch (std::exception &err) {
          std::cout << "\n\tError Must be integer Argument!";
          std::cout << "\n\t" << err.what() << std::endl;
          return 0;
        }
        break;

      default:
        abort();
    }
  }

  MPIData mpiData;
  Timing timing;
  Timing resetTime;
  Pointers pointer;
  timing.unitChoice = 1000000;     //1M
  resetTime.unitChoice = 1000000;  //1M

  std::vector<Timing> allTiming;
  allTiming.resize(user.partsToRunVector.size());

  MPI_Init(&argc, &argv);  //initialize communicator environment.

  if (help) {
    printHelp();
    MPI::Finalize();
    exit(0);
  }
  setupMPIAndVectors(mpiData, user);

  setupTime(timing, user);
  setupTime(resetTime, user);

  for (long unsigned int i = 0; i < user.partsToRunVector.size(); ++i) {
    if (user.partsToRunVector[i] == 1) {
      //setupTime(allTiming[i], user);
      //blockSendPart1(mpiData, allTiming[i], pointer, user);
      allTiming[i] = blockSendPart1(mpiData, timing, pointer, user);
      timing = resetTime;

    } else if (user.partsToRunVector[i] == 2) {
      //setupTime(allTiming[i], user);
      //blockSendPart2(mpiData, allTiming[i], pointer, user);
      allTiming[i] = blockSendPart2(mpiData, timing, pointer, user);
      timing = resetTime;

    } else if (user.partsToRunVector[i] == 3) {
      allTiming[i] = blockSendPart3(mpiData, timing, pointer, user);
      timing = resetTime;
      // } else if (user.partsToRunVector[i] == 4) {
      //   allTiming[i] = cudaTimePart4(timing, vect, dvect, size);

      // } else if (user.partsToRunVector[i] == 5) {
      //   allTiming[i] = cudaTimePart5(timing, vect, dvect, size);

    } else {
      std::cout << "\n\n\tError the User has not chose any number of Function!\n";
      break;
    }
  }

  if (!mpiData.rank)
    printTable(allTiming, printStander);

  MPI::Finalize();
  return 0;
}
const std::vector<int> chooseFunction(int toInteger) {
  std::vector<int> digits(0, 0);
  std::vector<int> ERROR(0, 0);

  int digit{1};

  while (toInteger > 0) {
    digit = toInteger % 10;
    if (digit > 7) {
      std::cout << "\n\tError Must be integer Argument <= " << toInteger << std::endl;
      return ERROR;
    }
    digits.push_back(digit);
    toInteger /= 10;
  }
  std::reverse(digits.begin(), digits.end());
  return digits;
}

std::pair<int, int> splitProcess(int works, int numberOfProcess) {
  std::pair<int, int> Return{0, 0};
  if (numberOfProcess > 1 && numberOfProcess <= works) {
    Return.first = works / (numberOfProcess - 1);   //number of cycle for each process.
    Return.second = works % (numberOfProcess - 1);  //extra cycle for process.
  } else {
    std::cout << "\tError Either No worker are found OR Number Processes Larger than Length!!!\n";
  }

  return Return;
}
const std::vector<int> numberDataSend(int numberOfProcess, std::pair<int, int> splitWorks) {
  std::vector<int> dataSend(numberOfProcess, splitWorks.first);
  dataSend[0] = 0;
  for (int i = 1; i < splitWorks.second + 1; i++)  //neglect root
  {
    dataSend[i] += 1;  //extra work for each first processes.
  }
  return dataSend;
}
const std::vector<int> displacmentData(int numberOfProcess,
                                       std::pair<int, int> splitWorks,
                                       const std::vector<int> &numberDataSend) {
  std::vector<int> displacment(numberOfProcess, splitWorks.first);

  displacment[0] = 0;
  displacment[1] = 0;  //start Here.

  for (int i = 2; i < numberOfProcess; i++)  //neglect root
  {
    displacment[i] = numberDataSend[i - 1] + displacment[i - 1];  //extra work for each first processes.
  }
  return displacment;
}
void randomGenerator(float *vect) {
  std::random_device rand;
  std::default_random_engine gener(rand());
  std::uniform_real_distribution<> dis(0., 1.);
  for (unsigned int i = 0; i < sizeVector; ++i) {
    vect[i] = dis(gener);
  }
}
void setupMPIAndVectors(MPIData &mpiData, UserChoises &user) {
  mpiData.num_procs = MPI::COMM_WORLD.Get_size();  //get total size of processes.
  mpiData.rank = MPI::COMM_WORLD.Get_rank();       //get each process number.

  user.sizeVectorBytes = sizeVector * sizeof(float);  //get size in byte for vectors.

  mpiData.mVect1 = (float *)malloc(user.sizeVectorBytes);  //initialize size.
  mpiData.mVect2 = (float *)malloc(user.sizeVectorBytes);
  mpiData.mVect3 = (float *)malloc(user.sizeVectorBytes);
  mpiData.mVectChecking = (float *)malloc(user.sizeVectorBytes);

  //mpiData.mVectWorker1 = (float*) malloc(user.sizeVectorBytes);
  //mpiData.mVectWorker2 = (float*) malloc(user.sizeVectorBytes);
  mpiData.mVectWorker3 = (float *)malloc(user.sizeVectorBytes);

  mpiData.workSplit = splitProcess(sizeVector, mpiData.num_procs);

  if (!mpiData.workSplit.first) {
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    exit(-1);
  }

  mpiData.numberToSend = numberDataSend(mpiData.num_procs, mpiData.workSplit);
  mpiData.displacement = displacmentData(mpiData.num_procs, mpiData.workSplit, mpiData.numberToSend);

  // mpiData.mVectWorker1.resize(mpiData.numberToSend[mpiData.rank]);  //Resizing each process with appropriate Receiving Data.
  // mpiData.mVectWorker2.resize(mpiData.numberToSend[mpiData.rank]);
  mpiData.mVectWorker1 = (float *)malloc(mpiData.numberToSend[mpiData.rank] * sizeof(float));
  mpiData.mVectWorker2 = (float *)malloc(mpiData.numberToSend[mpiData.rank] * sizeof(float));

  if (!mpiData.rank)  //Only for root
  {
    randomGenerator(mpiData.mVect1);  //generate random floating numbers from(0,1) Only in the root.
    randomGenerator(mpiData.mVect2);
    std::cout << "\n\tNumber of Processes " << mpiData.num_procs << std::endl;
    std::cout << "\tNumber of workSplit First " << mpiData.workSplit.first << std::endl;
    std::cout << "\tNumber of workSplit Second " << mpiData.workSplit.second << std::endl;
    std::cout << "\tTotal size of a Buffer " << user.sizeVectorBytes << " B" << std::endl;
  }
}
void setupTime(Timing &timing, UserChoises &user) {
  //Setup Verctors for Taking Average and Standard deviation
  timing.timeInputPreparationRoot.resize(average + user.extra);  //extra for saving the average.
  timing.timeInputPreparationHost.resize(average + user.extra);
  timing.timeOperationOnDeviceByRootHost.resize(average + user.extra);
  timing.timeOutputPreparationRoot.resize(average + user.extra);
  timing.timeOutputPreparationHost.resize(average + user.extra);
  timing.operationOnDeviceByDeviceAcc.resize(average + user.extra);
  timing.averageResults.resize(user.averageVectorSend);
}

void calculateTimeDuration(Timing &timing, int i, int &root) {
  if (!root) {
    timing.timeInputPreparationRoot[i] =
        (timing.inputPreparationRoot[1] - timing.inputPreparationRoot[0]);  //getting the time in microseconds
    timing.timeOperationOnDeviceByRootHost[i] = (timing.outputPreparationRoot[0] - timing.inputPreparationRoot[1]);
    timing.timeOutputPreparationRoot[i] = (timing.outputPreparationRoot[1] - timing.outputPreparationRoot[0]);
  } else {
    timing.timeInputPreparationHost[i] = (timing.inputPreparationHost[1] - timing.inputPreparationHost[0]);
    timing.timeOperationOnDeviceByRootHost[i] =
        (timing.operationOnDeviceByHost[1] -
         timing.operationOnDeviceByHost[0]);  //time taking for Device operation with respect of Host.
    cudaEventElapsedTime(&timing.operationOnDeviceByDevice,
                         timing.start,
                         timing.stop);  //get the time elapse in Device operation with device perspective.
    timing.operationOnDeviceByDeviceAcc[i] = (timing.operationOnDeviceByDevice * 1000);
    timing.timeOutputPreparationHost[i] = (timing.outputPreparationHost[1] - timing.outputPreparationHost[0]);
  }
}
void addVectorsHost(float *vect1, float *vect2, float *vect3) {
  for (unsigned int i = 0; i < sizeVector; ++i) {
    vect3[i] = vect2[i] + vect1[i];
  }
}
void cleanBuffer(float *vect) {
  for (unsigned int i = 0; i < sizeVector; ++i) {
    vect[i] = 0;
  }
}
bool checkingResultsPrintout(float *vectCpu, float *vectGpu) {
  float percent{0.0};
  float totalError{0.0};

  for (unsigned int j = 0; j < sizeVector; j++) {
    percent = ((vectCpu[j] - vectGpu[j]) / vectCpu[j]) * 100;
    totalError += percent;
  }
  if (totalError) {
    std::cout << "\n------------------------------------\n";
    std::cout << "| CpuSum | GpuSum | Error  | Error %| ";
    std::cout << "\n------------------------------------\n";
    //std::cout.precision(4);
    for (unsigned int j = 0; j < sizeVector; j++) {
      std::cout.flags(std::ios::fixed | std::ios::showpoint);
      std::cout.precision(4);
      std::cout << "| " << vectCpu[j] << " | " << vectGpu[j] << " | " << vectCpu[j] - vectGpu[j] << " | " << percent
                << " |\n";
    }
    std::cout << "-------------------------------------\n";
    std::cout << "-Total Error is " << totalError << std::endl;
    return false;
  }
  return true;
}
void calculateAverageDeviation(Timing &timing, int averg, int &root) {
  //Average
  for (int i = 0; i < averg; ++i) {
    if (!root) {
      timing.timeInputPreparationRoot[averg] += timing.timeInputPreparationRoot[i];
      timing.timeOperationOnDeviceByRootHost[averg] += timing.timeOperationOnDeviceByRootHost[i];
      timing.timeOutputPreparationRoot[averg] += timing.timeOutputPreparationRoot[i];
    } else {
      timing.timeInputPreparationHost[averg] += timing.timeInputPreparationHost[i];
      timing.timeOperationOnDeviceByRootHost[averg] += timing.timeOperationOnDeviceByRootHost[i];
      timing.timeOutputPreparationHost[averg] += timing.timeOutputPreparationHost[i];
      timing.operationOnDeviceByDeviceAcc[averg] += timing.operationOnDeviceByDeviceAcc[i];
    }
  }
  if (!root) {
    timing.timeInputPreparationRoot[averg] = timing.timeInputPreparationRoot[averg] / averg;
    timing.timeOperationOnDeviceByRootHost[averg] = timing.timeOperationOnDeviceByRootHost[averg] / averg;

    timing.timeOutputPreparationRoot[averg] = timing.timeOutputPreparationRoot[averg] / averg;

  } else {
    timing.timeInputPreparationHost[averg] = timing.timeInputPreparationHost[averg] / averg;

    timing.timeOperationOnDeviceByRootHost[averg] = timing.timeOperationOnDeviceByRootHost[averg] / averg;

    timing.timeOutputPreparationHost[averg] = timing.timeOutputPreparationHost[averg] / averg;

    timing.operationOnDeviceByDeviceAcc[averg] = (double)timing.operationOnDeviceByDeviceAcc[averg] / averg;
  }

  //Standard deviation
  for (int i = 0; i < averg; ++i) {
    if (!root) {
      timing.timeInputPreparationRoot[i] -= timing.timeInputPreparationRoot[averg];  //Take the different.
      timing.timeInputPreparationRoot[i] =
          timing.timeInputPreparationRoot[i] * timing.timeInputPreparationRoot[i];  // Square it.
      timing.timeInputPreparationRoot[averg + 1] +=
          timing.timeInputPreparationRoot[i];  //add them togather. averg+1 is location of the Deviation

      timing.timeOperationOnDeviceByRootHost[i] -= timing.timeOperationOnDeviceByRootHost[averg];
      timing.timeOperationOnDeviceByRootHost[i] *= timing.timeOperationOnDeviceByRootHost[i];
      timing.timeOperationOnDeviceByRootHost[averg + 1] += timing.timeOperationOnDeviceByRootHost[i];

      timing.timeOutputPreparationRoot[i] -= timing.timeOutputPreparationRoot[averg];
      timing.timeOutputPreparationRoot[i] *= timing.timeOutputPreparationRoot[i];
      timing.timeOutputPreparationRoot[averg + 1] += timing.timeOutputPreparationRoot[i];
    } else {
      timing.timeInputPreparationHost[i] -= timing.timeInputPreparationHost[averg];  //Take the different.
      timing.timeInputPreparationHost[i] =
          timing.timeInputPreparationHost[i] * timing.timeInputPreparationHost[i];  // Square it.
      timing.timeInputPreparationHost[averg + 1] +=
          timing.timeInputPreparationHost[i];  //add them togather. averg+1 is location of the Deviation

      timing.timeOperationOnDeviceByRootHost[i] -= timing.timeOperationOnDeviceByRootHost[averg];
      timing.timeOperationOnDeviceByRootHost[i] *= timing.timeOperationOnDeviceByRootHost[i];
      timing.timeOperationOnDeviceByRootHost[averg + 1] += timing.timeOperationOnDeviceByRootHost[i];

      timing.timeOutputPreparationHost[i] -= timing.timeOutputPreparationHost[averg];
      timing.timeOutputPreparationHost[i] *= timing.timeOutputPreparationHost[i];
      timing.timeOutputPreparationHost[averg + 1] += timing.timeOutputPreparationHost[i];

      timing.operationOnDeviceByDeviceAcc[i] -= timing.operationOnDeviceByDeviceAcc[averg];
      timing.operationOnDeviceByDeviceAcc[i] *= timing.operationOnDeviceByDeviceAcc[i];
      timing.operationOnDeviceByDeviceAcc[averg + 1] += timing.operationOnDeviceByDeviceAcc[i];
    }
  }

  if (!root) {
    timing.timeInputPreparationRoot[averg + 1] = timing.timeInputPreparationRoot[averg + 1] / averg;
    timing.timeInputPreparationRoot[averg + 1] = sqrt(timing.timeInputPreparationRoot[averg + 1]);

    timing.timeOperationOnDeviceByRootHost[averg + 1] = timing.timeOperationOnDeviceByRootHost[averg + 1] / averg;
    timing.timeOperationOnDeviceByRootHost[averg + 1] = sqrt(timing.timeOperationOnDeviceByRootHost[averg + 1]);

    timing.timeOutputPreparationRoot[averg + 1] = timing.timeOutputPreparationRoot[averg + 1] / averg;
    timing.timeOutputPreparationRoot[averg + 1] = sqrt(timing.timeOutputPreparationRoot[averg + 1]);

  } else {
    timing.timeInputPreparationHost[averg + 1] = timing.timeInputPreparationHost[averg + 1] / averg;  //*1000000
    timing.timeInputPreparationHost[averg + 1] = sqrt(timing.timeInputPreparationHost[averg + 1]);

    timing.timeOperationOnDeviceByRootHost[averg + 1] = timing.timeOperationOnDeviceByRootHost[averg + 1] / averg;
    timing.timeOperationOnDeviceByRootHost[averg + 1] = sqrt(timing.timeOperationOnDeviceByRootHost[averg + 1]);

    timing.timeOutputPreparationHost[averg + 1] = timing.timeOutputPreparationHost[averg + 1] / averg;
    timing.timeOutputPreparationHost[averg + 1] = sqrt(timing.timeOutputPreparationHost[averg + 1]);

    timing.operationOnDeviceByDeviceAcc[averg + 1] = (double)timing.operationOnDeviceByDeviceAcc[averg + 1] / averg;
    timing.operationOnDeviceByDeviceAcc[averg + 1] = sqrt(timing.operationOnDeviceByDeviceAcc[averg + 1]);
  }

  if (!root) {
    timing.timeInputPreparationRoot[averg] *= timing.unitChoice;
    timing.timeOperationOnDeviceByRootHost[averg] *= timing.unitChoice;
    timing.timeOutputPreparationRoot[averg] *= timing.unitChoice;

    timing.timeInputPreparationRoot[averg + 1] *= timing.unitChoice;
    timing.timeOperationOnDeviceByRootHost[averg + 1] *= timing.unitChoice;
    timing.timeOutputPreparationRoot[averg + 1] *= timing.unitChoice;
  } else {
    timing.timeInputPreparationHost[averg] *= timing.unitChoice;
    timing.timeOperationOnDeviceByRootHost[averg] *= timing.unitChoice;
    timing.timeOutputPreparationHost[averg] *= timing.unitChoice;

    timing.timeInputPreparationHost[averg + 1] *= timing.unitChoice;
    timing.timeOperationOnDeviceByRootHost[averg + 1] *= timing.unitChoice;
    timing.timeOutputPreparationHost[averg + 1] *= timing.unitChoice;
  }
}

bool sendAverageToRoot(Timing &timing, UserChoises &user, int &rank) {
  if (rank) {
    timing.averageResults[0] = timing.timeInputPreparationHost[average];
    timing.averageResults[1] = timing.timeInputPreparationHost[average + 1];  //Stander Deviation

    timing.averageResults[2] = timing.timeOperationOnDeviceByRootHost[average];
    timing.averageResults[3] = timing.timeOperationOnDeviceByRootHost[average + 1];

    timing.averageResults[4] = timing.timeOutputPreparationHost[average];
    timing.averageResults[5] = timing.timeOutputPreparationHost[average + 1];

    timing.averageResults[6] = timing.operationOnDeviceByDeviceAcc[average];
    timing.averageResults[7] = timing.operationOnDeviceByDeviceAcc[average + 1];

    MPI_Send(&timing.averageResults[0], user.averageVectorSend, MPI_FLOAT, user.root, 0, MPI_COMM_WORLD);

  } else if (!rank) {
    MPI_Recv(&timing.averageResults[0], user.averageVectorSend, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  return true;
}

Timing blockSendPart1(MPIData &mpiData, Timing &timing, Pointers &pointer, UserChoises &user) {
  cleanBuffer(mpiData.mVectWorker3);  //clear each value of vector's elements
  timing.partChosen = 1;

  if (mpiData.rank)  //Only for Workers
  {
    cudaCheck(cudaMalloc((void **)&pointer.dVect1,
                         user.sizeVectorBytes));  //allocate memory space for vector in the global memory of the Device.
    cudaCheck(cudaMalloc((void **)&pointer.dVect2, user.sizeVectorBytes));
    cudaCheck(cudaMalloc((void **)&pointer.dVect3, user.sizeVectorBytes));
  }
  ///////////////////////////// Start of Average ////////////////////////
  for (int a = 0; a <= average; ++a) {
    if (!mpiData.rank)  //Only for root
    {
      ////////////////////////////////// Input Prepation for Root //////////////////////////////////
      timing.inputPreparationRoot[0] = MPI_Wtime();
      for (int i = 1; i < mpiData.num_procs; ++i) {
        MPI_Send(&mpiData.mVect1[mpiData.displacement[i]],
                 mpiData.numberToSend[i],
                 MPI_FLOAT,
                 i,
                 0,
                 MPI_COMM_WORLD);  //Tag is 0
        MPI_Send(&mpiData.mVect2[mpiData.displacement[i]], mpiData.numberToSend[i], MPI_FLOAT, i, 0, MPI_COMM_WORLD);
      }
      timing.inputPreparationRoot[1] = MPI_Wtime();
      /////////////////////////////////////////////////////////////////////////////////////////////////
    }

    if (mpiData.rank)  //Only for Workers
    {
      ////////////////////////////////// Input Prepation for Host //////////////////////////////////
      MPI_Probe(user.root, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      timing.inputPreparationHost[0] = MPI_Wtime();
      MPI_Recv(&mpiData.mVectWorker1[0],
               mpiData.numberToSend[mpiData.rank],
               MPI_FLOAT,
               user.root,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&mpiData.mVectWorker2[0],
               mpiData.numberToSend[mpiData.rank],
               MPI_FLOAT,
               user.root,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      cudaCheck(cudaMemcpy(pointer.dVect1,
                           mpiData.mVectWorker1,
                           user.sizeVectorBytes,
                           cudaMemcpyHostToDevice));  //copy random vector from host to device.
      cudaCheck(cudaMemcpy(pointer.dVect2, mpiData.mVectWorker2, user.sizeVectorBytes, cudaMemcpyHostToDevice));

      timing.inputPreparationHost[1] = MPI_Wtime();
      ///////////////////////////////////////////////////////////////////////////////////////

      cudaCheck(cudaEventCreate(&timing.start));  //inialize Event.
      cudaCheck(cudaEventCreate(&timing.stop));

      ///////////////////////////// Operation on Device with respect of Host //////////////////

      int threads = 512;                                  //arbitrary number.
      int blocks = (sizeVector + threads - 1) / threads;  //get ceiling number of blocks.
      blocks = std::min(blocks, 8);  // Number 8 is least number can be got from lowest Nevedia GPUs.

      ////////////////////////// CAll Device Kernel //////////////////////////////////
      cudaCheck(cudaEventRecord(timing.start));
      timing.operationOnDeviceByHost[0] = MPI_Wtime();

      addVectorsGpu<<<blocks, threads>>>(pointer.dVect1,
                                         pointer.dVect2,
                                         pointer.dVect3,
                                         sizeVector,
                                         task);  //call device function to add two vectors and save into vect3Gpu.

      cudaCheck(cudaGetLastError());
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaEventRecord(timing.stop));

      timing.operationOnDeviceByHost[1] = MPI_Wtime();
      /////////////////////////////////////////////////////////////////////////////////////////////

      /////////////////////////////////// Output Prepation for the Host //////////////////////////////////////
      timing.outputPreparationHost[0] = MPI_Wtime();
      cudaCheck(cudaMemcpy(
          mpiData.mVectWorker3,
          pointer.dVect3,
          user.sizeVectorBytes,
          cudaMemcpyDeviceToHost));  //copy summing result vector from Device to Host.// Try_Regist(3) delete this

      MPI_Send(&mpiData.mVectWorker3[0],
               mpiData.numberToSend[mpiData.rank],
               MPI_FLOAT,
               user.root,
               0,
               MPI_COMM_WORLD);  //Tag is 0
      timing.outputPreparationHost[1] = MPI_Wtime();
      ////////////////////////////////////////////////////////////////////////////////////////////////
    }

    if (!mpiData.rank)  //Only for root
    {
      /////////////////////////////////// Output Prepation for the Root //////////////////////////////////////
      MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      timing.outputPreparationRoot[0] = MPI_Wtime();
      //MPI probe
      for (int i = 1; i < mpiData.num_procs; i++) {
        MPI_Recv(&mpiData.mVectWorker3[mpiData.displacement[i]],
                 mpiData.numberToSend[i],
                 MPI_FLOAT,
                 i,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      }
      timing.outputPreparationRoot[1] = MPI_Wtime();
      ////////////////////////////////////////////////////////////////////////////////////////////////
    }

    if (a > 0)
      calculateTimeDuration(timing, a - 1, mpiData.rank);

    if (mpiData.rank) {
      cudaCheck(cudaEventDestroy(timing.start));
      cudaCheck(cudaEventDestroy(timing.stop));
    }
  }
  ///////////////////////////// End of Average ////////////////////////
  if (mpiData.rank) {
    cudaCheck(cudaFree(pointer.dVect1));
    cudaCheck(cudaFree(pointer.dVect2));
    cudaCheck(cudaFree(pointer.dVect3));
  }
  ///
  bool test = 0;
  if (!mpiData.rank)  //Only for root
  {
    addVectorsHost(mpiData.mVect1, mpiData.mVect2, mpiData.mVectChecking);  //Host is adding vectors too.
    test = checkingResultsPrintout(mpiData.mVectChecking,
                                   mpiData.mVectWorker3);  //Checking the results, if error then Print out to the user.
    if (!test)
      exit(-1);
  }

  calculateAverageDeviation(timing, average, mpiData.rank);
  test = sendAverageToRoot(timing, user, mpiData.rank);
  if (test && !mpiData.rank) {
    if (saveFile) {
      test = saveToFile("dataPart1", timing);

      if (test)
        std::cout << "Done Part " << timing.partChosen << " And File saved" << std::endl;
      else
        std::cout << "Error Saving File!!" << std::endl;
    }
    std::cout << "Done Part " << timing.partChosen << std::endl;
  }
  return timing;
}

Timing blockSendPart2(MPIData &mpiData, Timing &timing, Pointers &pointer, UserChoises &user) {
  cleanBuffer(mpiData.mVectWorker3);  //clear each value of vector's elements
  timing.partChosen = 2;

  if (mpiData.rank)  //Only for Workers
  {
    cudaCheck(cudaMallocHost((void **)&pointer.vect1, user.sizeVectorBytes));  //allocate Pinned memory on the Host.
    cudaCheck(cudaMallocHost((void **)&pointer.vect2, user.sizeVectorBytes));
    cudaCheck(cudaMallocHost((void **)&pointer.vect3, user.sizeVectorBytes));
    cudaCheck(cudaMalloc((void **)&pointer.dVect1,
                         user.sizeVectorBytes));  //allocate memory space for vector in the global memory of the Device.
    cudaCheck(cudaMalloc((void **)&pointer.dVect2, user.sizeVectorBytes));
    cudaCheck(cudaMalloc((void **)&pointer.dVect3, user.sizeVectorBytes));
  }
  ///////////////////////////// Start of Average ////////////////////////
  for (int a = 0; a <= average; ++a) {
    if (!mpiData.rank)  //Only for root
    {
      ////////////////////////////////// Input Prepation for Root //////////////////////////////////
      timing.inputPreparationRoot[0] = MPI_Wtime();
      for (int i = 1; i < mpiData.num_procs; ++i) {
        MPI_Send(&mpiData.mVect1[mpiData.displacement[i]],
                 mpiData.numberToSend[i],
                 MPI_FLOAT,
                 i,
                 0,
                 MPI_COMM_WORLD);  //Tag is 0
        MPI_Send(&mpiData.mVect2[mpiData.displacement[i]], mpiData.numberToSend[i], MPI_FLOAT, i, 0, MPI_COMM_WORLD);
      }
      timing.inputPreparationRoot[1] = MPI_Wtime();
      /////////////////////////////////////////////////////////////////////////////////////////////////
    }

    if (mpiData.rank)  //Only for Workers
    {
      ////////////////////////////////// Input Prepation for Host //////////////////////////////////
      MPI_Probe(user.root, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      timing.inputPreparationHost[0] = MPI_Wtime();

      MPI_Recv(&pointer.vect1[0],
               mpiData.numberToSend[mpiData.rank],
               MPI_FLOAT,
               user.root,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&pointer.vect2[0],
               mpiData.numberToSend[mpiData.rank],
               MPI_FLOAT,
               user.root,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      cudaCheck(cudaMemcpy(pointer.dVect1,
                           pointer.vect1,
                           user.sizeVectorBytes,
                           cudaMemcpyHostToDevice));  //copy random vector from host to device.
      cudaCheck(cudaMemcpy(pointer.dVect2, pointer.vect2, user.sizeVectorBytes, cudaMemcpyHostToDevice));

      timing.inputPreparationHost[1] = MPI_Wtime();
      ///////////////////////////////////////////////////////////////////////////////////////

      cudaCheck(cudaEventCreate(&timing.start));  //inialize Event.
      cudaCheck(cudaEventCreate(&timing.stop));

      ///////////////////////////// Operation on Device with respect of Host //////////////////

      int threads = 512;                                  //arbitrary number.
      int blocks = (sizeVector + threads - 1) / threads;  //get ceiling number of blocks.
      blocks = std::min(blocks, 8);  // Number 8 is least number can be got from lowest Nevedia GPUs.

      ////////////////////////// CAll Device Kernel //////////////////////////////////
      cudaCheck(cudaEventRecord(timing.start));
      timing.operationOnDeviceByHost[0] = MPI_Wtime();

      addVectorsGpu<<<blocks, threads>>>(pointer.dVect1,
                                         pointer.dVect2,
                                         pointer.dVect3,
                                         sizeVector,
                                         task);  //call device function to add two vectors and save into vect3Gpu.

      cudaCheck(cudaGetLastError());
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaEventRecord(timing.stop));

      timing.operationOnDeviceByHost[1] = MPI_Wtime();
      /////////////////////////////////////////////////////////////////////////////////////////////

      /////////////////////////////////// Output Prepation for the Host //////////////////////////////////////
      timing.outputPreparationHost[0] = MPI_Wtime();

      cudaCheck(cudaMemcpy(
          pointer.vect3,
          pointer.dVect3,
          user.sizeVectorBytes,
          cudaMemcpyDeviceToHost));  //copy summing result vector from Device to Host.// Try_Regist(3) delete this

      MPI_Send(&pointer.vect3[0],
               mpiData.numberToSend[mpiData.rank],
               MPI_FLOAT,
               user.root,
               0,
               MPI_COMM_WORLD);  //Tag is 0

      timing.outputPreparationHost[1] = MPI_Wtime();
      ////////////////////////////////////////////////////////////////////////////////////////////////
    }

    if (!mpiData.rank)  //Only for root
    {
      /////////////////////////////////// Output Prepation for the Root //////////////////////////////////////
      MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      timing.outputPreparationRoot[0] = MPI_Wtime();
      //MPI probe
      for (int i = 1; i < mpiData.num_procs; i++) {
        MPI_Recv(&mpiData.mVectWorker3[mpiData.displacement[i]],
                 mpiData.numberToSend[i],
                 MPI_FLOAT,
                 i,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      }
      timing.outputPreparationRoot[1] = MPI_Wtime();
      ////////////////////////////////////////////////////////////////////////////////////////////////
    }

    if (a > 0)
      calculateTimeDuration(timing, a - 1, mpiData.rank);

    if (mpiData.rank) {
      cudaCheck(cudaEventDestroy(timing.start));
      cudaCheck(cudaEventDestroy(timing.stop));
    }
  }
  ///////////////////////////// End of Average ////////////////////////
  if (mpiData.rank) {
    cudaCheck(cudaFreeHost(pointer.vect1));
    cudaCheck(cudaFreeHost(pointer.vect2));
    cudaCheck(cudaFreeHost(pointer.vect3));
    cudaCheck(cudaFree(pointer.dVect1));
    cudaCheck(cudaFree(pointer.dVect2));
    cudaCheck(cudaFree(pointer.dVect3));
  }

  bool test = 0;
  if (!mpiData.rank)  //Only for root
  {
    addVectorsHost(mpiData.mVect1, mpiData.mVect2, mpiData.mVectChecking);  //Host is adding vectors too.
    test = checkingResultsPrintout(mpiData.mVectChecking,
                                   mpiData.mVectWorker3);  //Checking the results, if error then Print out to the user.
    if (!test)
      exit(-1);
  }

  calculateAverageDeviation(timing, average, mpiData.rank);
  test = sendAverageToRoot(timing, user, mpiData.rank);
  if (test && !mpiData.rank) {
    if (saveFile) {
      test = saveToFile("dataPart2", timing);

      if (test)
        std::cout << "Done Part " << timing.partChosen << " And File saved" << std::endl;
      else
        std::cout << "Error Saving File!!" << std::endl;
    }
    std::cout << "Done Part " << timing.partChosen << std::endl;
  }
  return timing;
}

Timing blockSendPart3(MPIData &mpiData, Timing &timing, Pointers &pointer, UserChoises &user) {
  cleanBuffer(mpiData.mVectWorker3);  //clear each value of vector's elements
  timing.partChosen = 3;

  if (mpiData.rank)  //Only for Workers
  {
    cudaCheck(cudaMalloc((void **)&pointer.dVect1,
                         user.sizeVectorBytes));  //allocate memory space for vector in the global memory of the Device.
    cudaCheck(cudaMalloc((void **)&pointer.dVect2, user.sizeVectorBytes));
    cudaCheck(cudaMalloc((void **)&pointer.dVect3, user.sizeVectorBytes));
  }
  ///////////////////////////// Start of Average ////////////////////////
  for (int a = 0; a <= average; ++a) {
    if (!mpiData.rank)  //Only for root
    {
      ////////////////////////////////// Input Prepation for Root //////////////////////////////////
      timing.inputPreparationRoot[0] = MPI_Wtime();
      for (int i = 1; i < mpiData.num_procs; ++i) {
        MPI_Send(&mpiData.mVect1[mpiData.displacement[i]],
                 mpiData.numberToSend[i],
                 MPI_FLOAT,
                 i,
                 0,
                 MPI_COMM_WORLD);  //Tag is 0
        MPI_Send(&mpiData.mVect2[mpiData.displacement[i]], mpiData.numberToSend[i], MPI_FLOAT, i, 0, MPI_COMM_WORLD);
      }
      timing.inputPreparationRoot[1] = MPI_Wtime();
      /////////////////////////////////////////////////////////////////////////////////////////////////
    }

    if (mpiData.rank)  //Only for Workers
    {
      ////////////////////////////////// Input Prepation for Host //////////////////////////////////
      MPI_Probe(user.root, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      timing.inputPreparationHost[0] = MPI_Wtime();
      MPI_Recv(&pointer.dVect1[0],
               mpiData.numberToSend[mpiData.rank],
               MPI_FLOAT,
               user.root,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&pointer.dVect2[0],
               mpiData.numberToSend[mpiData.rank],
               MPI_FLOAT,
               user.root,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      // cudaCheck(cudaMemcpy(pointer.dVect1, mpiData.mVectWorker1, user.sizeVectorBytes, cudaMemcpyHostToDevice));  //copy random vector from host to device.
      // cudaCheck(cudaMemcpy(pointer.dVect2, mpiData.mVectWorker2, user.sizeVectorBytes, cudaMemcpyHostToDevice));

      timing.inputPreparationHost[1] = MPI_Wtime();
      ///////////////////////////////////////////////////////////////////////////////////////

      cudaCheck(cudaEventCreate(&timing.start));  //inialize Event.
      cudaCheck(cudaEventCreate(&timing.stop));

      ///////////////////////////// Operation on Device with respect of Host //////////////////

      int threads = 512;                                  //arbitrary number.
      int blocks = (sizeVector + threads - 1) / threads;  //get ceiling number of blocks.
      blocks = std::min(blocks, 8);  // Number 8 is least number can be got from lowest Nevedia GPUs.

      ////////////////////////// CAll Device Kernel //////////////////////////////////
      cudaCheck(cudaEventRecord(timing.start));
      timing.operationOnDeviceByHost[0] = MPI_Wtime();

      addVectorsGpu<<<blocks, threads>>>(pointer.dVect1,
                                         pointer.dVect2,
                                         pointer.dVect3,
                                         sizeVector,
                                         task);  //call device function to add two vectors and save into vect3Gpu.

      cudaCheck(cudaGetLastError());
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaEventRecord(timing.stop));

      timing.operationOnDeviceByHost[1] = MPI_Wtime();
      /////////////////////////////////////////////////////////////////////////////////////////////

      /////////////////////////////////// Output Prepation for the Host //////////////////////////////////////
      timing.outputPreparationHost[0] = MPI_Wtime();
      //cudaCheck(cudaMemcpy(mpiData.mVectWorker3,pointer.dVect3,user.sizeVectorBytes,cudaMemcpyDeviceToHost));  //copy summing result vector from Device to Host.// Try_Regist(3) delete this

      MPI_Send(&pointer.dVect3[0],
               mpiData.numberToSend[mpiData.rank],
               MPI_FLOAT,
               user.root,
               0,
               MPI_COMM_WORLD);  //Tag is 0
      timing.outputPreparationHost[1] = MPI_Wtime();
      ////////////////////////////////////////////////////////////////////////////////////////////////
    }

    if (!mpiData.rank)  //Only for root
    {
      /////////////////////////////////// Output Prepation for the Root //////////////////////////////////////
      MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      timing.outputPreparationRoot[0] = MPI_Wtime();
      //MPI probe
      for (int i = 1; i < mpiData.num_procs; i++) {
        MPI_Recv(&mpiData.mVectWorker3[mpiData.displacement[i]],
                 mpiData.numberToSend[i],
                 MPI_FLOAT,
                 i,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      }
      timing.outputPreparationRoot[1] = MPI_Wtime();
      ////////////////////////////////////////////////////////////////////////////////////////////////
    }

    if (a > 0)
      calculateTimeDuration(timing, a - 1, mpiData.rank);

    if (mpiData.rank) {
      cudaCheck(cudaEventDestroy(timing.start));
      cudaCheck(cudaEventDestroy(timing.stop));
    }
  }
  ///////////////////////////// End of Average ////////////////////////
  if (mpiData.rank) {
    cudaCheck(cudaFree(pointer.dVect1));
    cudaCheck(cudaFree(pointer.dVect2));
    cudaCheck(cudaFree(pointer.dVect3));
  }
  ///
  bool test = 0;
  if (!mpiData.rank)  //Only for root
  {
    addVectorsHost(mpiData.mVect1, mpiData.mVect2, mpiData.mVectChecking);  //Host is adding vectors too.
    test = checkingResultsPrintout(mpiData.mVectChecking,
                                   mpiData.mVectWorker3);  //Checking the results, if error then Print out to the user.
    if (!test)
      exit(-1);
  }

  calculateAverageDeviation(timing, average, mpiData.rank);
  test = sendAverageToRoot(timing, user, mpiData.rank);
  if (test && !mpiData.rank) {
    if (saveFile) {
      test = saveToFile("dataPart3", timing);

      if (test)
        std::cout << "Done Part " << timing.partChosen << " And File saved" << std::endl;
      else
        std::cout << "Error Saving File!!" << std::endl;
    }
    std::cout << "Done Part " << timing.partChosen << std::endl;
  }
  return timing;
}

void printTable(std::vector<Timing> &timing, bool standerDeviationPrint) {
  const std::string inPrepatRoot = " Duration Time Read Input Prepations On Root ";
  const std::string inPrepatHost = " Duration Time Read Input Prepations On Host ";
  const std::string timeCpuR = " Duration Time operation on Root point View  ";
  const std::string timeCpu = " Duration Time operation on Host point View  ";
  const std::string timeGpu = " Duration Time operation on Device point View";
  const std::string outPrepatRoot = " Duration Time Read Output Prepations On Root";
  const std::string outPrepatHost = " Duration Time Read Output Prepations On Host";

  const std::string averageTime = " AverTime ";
  const std::string standerDeviation = " StDeviation ";
  const std::string nameTiming = " Name Timing ";
  const std::string partsNumberall = "Part ";

  int totalFix = 0;

  if (standerDeviationPrint) {
    totalFix = timeGpu.size() + timing.size() * (averageTime.size() + standerDeviation.size() + 3);
  } else {
    totalFix = timeGpu.size() + timing.size() * (averageTime.size() + 3);
  }

  std::cout.flags(std::ios::fixed | std::ios::showpoint);
  std::cout.precision(4);

  std::cout << '\n';
  std::cout.width(totalFix);
  std::cout.fill('-');
  std::cout << '-' << '\n';
  std::cout.fill(' ');

  std::cout << "|";
  std::cout.width((timeGpu.size() - nameTiming.size()) / 2);
  std::cout.fill(' ');
  std::cout << " ";
  std::cout << nameTiming;
  std::cout.width((timeGpu.size() - nameTiming.size()) / 2);
  std::cout.fill(' ');
  std::cout << " ";
  std::cout << "  |";

  for (unsigned int i = 0; i < timing.size(); ++i) {
    if (standerDeviationPrint) {
      std::cout.width(((averageTime.size() + standerDeviation.size()) - partsNumberall.size() + 1) / 2);
    }  //9
    else {
      std::cout.width(((averageTime.size()) - partsNumberall.size()) / 2);
    }  //2

    std::cout << " ";
    std::cout << partsNumberall << timing[i].partChosen;

    if (standerDeviationPrint) {
      std::cout.width(((averageTime.size() + standerDeviation.size()) - partsNumberall.size() + 1) / 2);
    }  //9
    else {
      std::cout.width(((averageTime.size()) - partsNumberall.size()) / 2);
    }
    //2
    std::cout << " ";
    std::cout << "|";
  }

  std::cout << '\n';
  std::cout << "|";
  std::cout.width(inPrepatHost.size() + 3);
  std::cout.fill(' ');
  std::cout << "|";

  for (unsigned int i = 0; i < timing.size(); ++i) {
    std::cout << averageTime;
    std::cout << "|";
    if (standerDeviationPrint) {
      std::cout << standerDeviation;
      std::cout << "|";
    }
  }

  newLineTitle(totalFix, inPrepatRoot);
  printResultEach(timing, 1, standerDeviationPrint);

  newLineTitle(totalFix, inPrepatHost);
  printResultEach(timing, 2, standerDeviationPrint);

  newLineTitle(totalFix, timeCpuR);
  printResultEach(timing, 3, standerDeviationPrint);

  newLineTitle(totalFix, timeCpu);
  printResultEach(timing, 4, standerDeviationPrint);

  newLineTitle(totalFix, timeGpu);
  printResultEach(timing, 5, standerDeviationPrint);

  newLineTitle(totalFix, outPrepatRoot);
  printResultEach(timing, 6, standerDeviationPrint);

  newLineTitle(totalFix, outPrepatHost);
  printResultEach(timing, 7, standerDeviationPrint);

  std::cout << '\n';
  std::cout.width(totalFix);
  std::cout.fill('-');
  std::cout << '-' << '\n';
  std::cout.fill(' ');
}
int getNumberofDigits(double number) { return ((int)log10(number) + 1) + 4; }
void newLineTitle(int line, const std::string &title) {
  std::cout << '\n';
  std::cout.width(line);
  std::cout.fill('-');
  std::cout << '-' << '\n';
  std::cout.fill(' ');

  std::cout << "| ";
  std::cout << title;
  std::cout << " |";
}
void printResultEach(std::vector<Timing> &timing, int type, bool standerDeviationPrint) {
  int averageTimeWidth = 10;
  int standerDeviationWidth = 13;

  for (unsigned int i = 0; i < timing.size(); ++i) {
    if (type == 1) {
      std::cout.width(averageTimeWidth);
      std::cout.fill(' ');
      std::cout << timing[i].timeInputPreparationRoot[average];
      std::cout << "|";
      if (standerDeviationPrint) {
        std::cout.width(standerDeviationWidth);
        std::cout.fill(' ');
        std::cout << timing[i].timeInputPreparationRoot[average + 1];
        std::cout << "|";
      }
    } else if (type == 2) {
      std::cout.width(averageTimeWidth);
      std::cout.fill(' ');
      std::cout << timing[i].averageResults[0];
      std::cout << "|";
      if (standerDeviationPrint) {
        std::cout.width(standerDeviationWidth);
        std::cout.fill(' ');
        std::cout << timing[i].averageResults[1];
        std::cout << "|";
      }
    } else if (type == 3) {
      std::cout.width(averageTimeWidth);
      std::cout.fill(' ');
      std::cout << timing[i].timeOperationOnDeviceByRootHost[average];
      std::cout << "|";
      if (standerDeviationPrint) {
        std::cout.width(standerDeviationWidth);
        std::cout.fill(' ');
        std::cout << timing[i].timeOperationOnDeviceByRootHost[average + 1];
        std::cout << "|";
      }
    } else if (type == 4) {
      std::cout.width(averageTimeWidth);
      std::cout.fill(' ');
      std::cout << timing[i].averageResults[2];
      std::cout << "|";
      if (standerDeviationPrint) {
        std::cout.width(standerDeviationWidth);
        std::cout.fill(' ');
        std::cout << timing[i].averageResults[3];
        std::cout << "|";
      }
    } else if (type == 5) {
      std::cout.width(averageTimeWidth);
      std::cout.fill(' ');
      std::cout << timing[i].averageResults[6];
      std::cout << "|";
      if (standerDeviationPrint) {
        std::cout.width(standerDeviationWidth);
        std::cout.fill(' ');
        std::cout << timing[i].averageResults[7];
        std::cout << "|";
      }
    } else if (type == 6) {
      std::cout.width(averageTimeWidth);
      std::cout.fill(' ');
      std::cout << timing[i].timeOutputPreparationRoot[average];
      std::cout << "|";
      if (standerDeviationPrint) {
        std::cout.width(standerDeviationWidth);
        std::cout.fill(' ');
        std::cout << timing[i].timeOutputPreparationRoot[average + 1];
        std::cout << "|";
      }
    } else if (type == 7) {
      std::cout.width(averageTimeWidth);
      std::cout.fill(' ');
      std::cout << timing[i].averageResults[4];
      std::cout << "|";
      if (standerDeviationPrint) {
        std::cout.width(standerDeviationWidth);
        std::cout.fill(' ');
        std::cout << timing[i].averageResults[5];
        std::cout << "|";
      }
    }
  }
}
bool saveToFile(const std::string &name, const Timing &timing) {
  std::ofstream file(name + ".txt", std::ios::out | std::ios::app);

  if (!file.is_open()) {
    std::cout << "\nCannot open File nor Create File!" << std::endl;
    return 0;
  }

  file << sizeVector << std::endl;
  file << average << std::endl;
  file << task << std::endl;
  file << timing.timeInputPreparationRoot[average] << " " << timing.timeInputPreparationRoot[average + 1] << std::endl;
  file << timing.averageResults[0] << " " << timing.averageResults[1] << std::endl;
  file << timing.timeOperationOnDeviceByRootHost[average] << " " << timing.timeOperationOnDeviceByRootHost[average + 1]
       << std::endl;
  file << timing.averageResults[2] << " " << timing.averageResults[3] << std::endl;
  file << timing.averageResults[6] << " " << timing.averageResults[7] << std::endl;
  file << timing.timeOutputPreparationRoot[average] << " " << timing.timeOutputPreparationRoot[average + 1]
       << std::endl;
  file << timing.averageResults[4] << " " << timing.averageResults[5] << std::endl;

  file.close();
  if (!file.good()) {
    std::cout << "\n*ERROR While Writing The " + name + " file!!" << std::endl;
    return 0;
  }
  return 1;
}
void printHelp(void) {
  int rank = MPI::COMM_WORLD.Get_rank();
  if (!rank) {
    std::cout << "\n\n\t**************************************\n";
    std::cout << "\t* This is a Help for Command Opitions*";
    std::cout << "\n\t**************************************\n";
    std::cout << "\n\tYou as a user, can choose two ways to run the program:\n";
    std::cout << "\n\t1) mpirun -np <number of Process/ors> -s <size of Vector> -t <number of task> -a <average size> "
                 "-p <part to run>\n";
    std::cout << "\n\t2) cmsenv_mpirun -np <number of Process/ors> -s <size of Vector> -t <number of task> -a <average "
                 "size> -p <part to run>\n";
    std::cout << "\n\t[-np] is for number of processes or processors that you would like to run.";
    std::cout
        << "\n\t[-s] is the size of vector that you would like to send, the type is float and there are two vectors.";
    std::cout << "\n\t[-t] is the number of repeating of task on the Device(GPU) side.";
    std::cout << "\n\t[-a] is the number of repeating the part that user has chosen.";
    std::cout << "\n\t[-p] is the choice of what part to run in the program.";
    std::cout << "\n\t[-q] is to print Stander Deviation.";
    std::cout << "\n\t[-f] is to save the results into a file for each part.";
    std::cout << "\n\n\tExample for only local Machine: ";
    std::cout << "\n\tcmsenv_mpirun -np 2 mpiCudaGeneric -p1 -s200 -t1 -a1\n";
    std::cout << "\n\tExample for two Machines connected: ";
    std::cout
        << "\n\tcmsenv_mpirun -H <machine Name as Root>,<machine Name as Host> -np 2 mpiCudaGeneric -p1 -s200 -t1 -a1";
    std::cout << "\n\tExample for two Machines connected Using ucx: ";
    std::cout << "\n\tcmsenv_mpirun -H <machine Name as Root>,<machine Name as Host> -np 2 -mca pml ucx -- "
                 "mpiCudaGeneric -p1 -s200 -t1 -a1";
    std::cout << "\n\n\tFor the Parts, we have in this program 4 Parts:";
    std::cout << "\n\t1)The Root, who does not have a GPU, using MPI Blocking send and receive to Host, The Host is "
                 "who have a GPU, then Host:";
    std::cout << "\n\t  uses cudaMalloc and copies the receiving values to GPU side. Next, the GPU does the compuation";
    std::cout << "\n\t  Finaly, the Host copies the results from GPU, sends them back to The Root using MPI Blocking "
                 "Send.\n\n";
  }
}
