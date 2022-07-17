#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <utility>
#include <chrono>  //Time
#include <cuda.h>
#include <thrust/device_vector.h>
#include <unistd.h>
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

//Global Varaibles
int sizeOfVector = 200;             //default size.
int averg = 10;                     //default average.
int extra = 2;                      //extra length in vectors for calculation.
int task = 100;                     //default number of task.
int part = 1;                       //default for user's choice of part.
int saveFile = 0;                   //default for saving results into a file.
int printStander = 0;               //default for printing stander deviation.
std::vector<int> partNumber(1, 1);  //vector for user's choice of part.

// Data Structur For Times
struct Timing {
  int partChosen;
  std::chrono::steady_clock::time_point copyToDevice[2];             // get time points from start and end.
  std::chrono::steady_clock::time_point operationOnDeviceByHost[2];  //get time duration in Device with Host perspective.
  std::chrono::steady_clock::time_point copyToHost[2];

  std::vector<std::chrono::duration<double, std::micro>> timeCopyToDevice;  //Save the Duration in Microsecond.
  std::vector<std::chrono::duration<double, std::micro>> timeOperationOnDeviceByHost;
  std::vector<std::chrono::duration<double, std::micro>> timeCopyToHost;

  cudaEvent_t start, stop;                          //get time points in Device.
  float operationOnDeviceByDevice = 0;              //get time duration in Device with device perspective.
  std::vector<float> operationOnDeviceByDeviceAcc;  //get accumulating time duration in Device with device perspective.
};

// Data Structure For Vectors
struct Vectors {
  std::vector<float> vect1;  //create vector.
  std::vector<float> vect2;
  std::vector<float> vect3Cpu;  //this is only for Host to verify.
  std::vector<float> vect3Gpu;  //this is only for Device.
};

//Data Structure for Pointers
struct Pointers {
  float *dVect1;
  float *dVect2;
  float *dVect3;

  float *dVect1Extra;
  float *dVect2Extra;
  float *dVect3Extra;
};

//called in the Host (CPU) and excuted in the Device (GPU)
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

//__host__//called in the Host (CPU) and executed in the host.
void addVectorsCpu(std::vector<float> &vect1,
                   std::vector<float> &vect2,
                   std::vector<float> &vect3);  //add two vectors and save the result into a third vector.

void randomGenerator(std::vector<float> &vect);  //generate uniform random numbers.

bool checkingResultsPrintout(std::vector<float> &vectCpu, std::vector<float> &vectGpu);  //printout the results.

void calculateTimeDuration(Timing &timing, int i);  //get Duration time for each cycle.

void calculateAverageDeviation(Timing &timing);  //Calculate Average and Standard deviation.

bool saveToFile(const std::string &name, const Timing &timing);

const std::vector<int> chooseFunction(int toInteger);  //Convert integers to a vector.

Timing cudaTimePart0(Timing &timing, Vectors &vect, Pointers &dvect, int size);  //Part 0 of Cuda measurement time.

Timing cudaTimePart1(
    Timing &timing, Vectors &vect, Pointers &dvect, int size, int startSave);  //PArt 1 of Cuda measurement time.

Timing cudaTimePart2(Timing &timing, Vectors &vect, int size);  //Part 2 of Cuda measurement time.

Timing cudaTimePart3(Timing &timing, Vectors &vect, Pointers &dvect, int size);  //Part 3 of Cuda measurement time.

Timing cudaTimePart4(Timing &timing, Vectors &vect, Pointers &dvect, int size);  //Part 4 of Cuda measurement time.

Timing cudaTimePart5(Timing &timing, Vectors &vect, Pointers &dvect, int size);  //Part 5 of Cuda measurement time.

void printoutAll(std::vector<Timing> &timing, bool standerDeviationPrint);

int getNumberofDigits(double number);

void newLineTitle(int line, const std::string &title);

void printResultEach(std::vector<Timing> &timing, int type, bool standerDeviationPrint);

int main(int argc, char *argv[]) {
  cms::cudatest::requireDevices();
  int c;  //to get parameters from user.
  while ((c = getopt(argc, argv, "s:a:t:p:fq")) != -1) {
    switch (c) {
      case 's':
        try {
          sizeOfVector = std::stoll(optarg, nullptr, 0);
        } catch (std::exception &err) {
          std::cout << "\n\tError Must be integer Argument!";
          std::cout << "\n\t" << err.what() << std::endl;
          return 0;
        }
        break;
      case 'a':
        try {
          averg = std::stoll(optarg, nullptr, 0);

        } catch (std::exception &err) {
          std::cout << "\n\tError Must be integer Argument!";
          std::cout << "\n\t" << err.what() << std::endl;
          return 0;
        }
        break;
      case 't':
        try {
          task = std::stoll(optarg, nullptr, 0);
          std::cout << "\nNumber of repeated Task is " << task << std::endl;
        } catch (std::exception &err) {
          std::cout << "\n\tError Must be integer Argument!";
          std::cout << "\n\t" << err.what() << std::endl;
          return 0;
        }
        break;
      case 'p':
        try {
          part = std::stoll(optarg, nullptr, 0);
          partNumber = chooseFunction(part);
          std::cout << "\nyou have chosen Part ";
          for (unsigned int j = 0; j < partNumber.size(); ++j) {
            std::cout << partNumber[j] << " ,";
          }
          std::cout << "\n";
        } catch (std::exception &err) {
          std::cout << "\n\tError Must be integer Argument!";
          std::cout << "\n\t" << err.what() << std::endl;
          return 0;
        }
        break;
      case 'f':
        try {
          saveFile = 1;
          std::cout << "\nyou have chosen to save file." << std::endl;
        } catch (std::exception &err) {
          std::cout << "\n\tError Must be integer Argument!";
          std::cout << "\n\t" << err.what() << std::endl;
          return 0;
        }
        break;
      case 'q':
        try {
          printStander = 1;
          std::cout << "\nyou have chosen to print stander Deviation." << std::endl;
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
  int size = sizeOfVector * sizeof(float);  //get size in byte for vectors.
  int startSave = 0;                        // to worm up GPU.
  std::vector<Timing> allTiming;
  allTiming.resize(partNumber.size());

  Timing timing;     //times Variables.
  Timing resetTime;  //To reset timing object.
  Vectors vect;      //Vectros variables.
  Pointers dvect;    //Pointers for device vectors.

  //Initialize vectors:
  vect.vect1.resize(sizeOfVector);
  vect.vect2.resize(sizeOfVector);
  vect.vect3Cpu.resize(sizeOfVector);
  vect.vect3Gpu.resize(sizeOfVector);

  //Setup Verctors for Taking Average and Standard deviation
  timing.timeCopyToDevice.resize(averg + extra);  //extra for saving the average.
  timing.timeOperationOnDeviceByHost.resize(averg + extra);
  timing.timeCopyToHost.resize(averg + extra);
  timing.operationOnDeviceByDeviceAcc.resize(averg + extra);

  //Setup Verctors for reseting timing.
  resetTime.timeCopyToDevice.resize(averg + extra);  //extra for saving the average.
  resetTime.timeOperationOnDeviceByHost.resize(averg + extra);
  resetTime.timeCopyToHost.resize(averg + extra);
  resetTime.operationOnDeviceByDeviceAcc.resize(averg + extra);

  //generate random numbers.
  randomGenerator(vect.vect1);
  randomGenerator(vect.vect2);

  for (unsigned int i = 0; i < partNumber.size(); ++i) {
    if (partNumber[i] == 6) {
      allTiming[i] = cudaTimePart0(timing, vect, dvect, size);
      timing = resetTime;  //reset timing.
    } else if (partNumber[i] == 1) {
      allTiming[i] = cudaTimePart1(timing, vect, dvect, size, startSave++);
      timing = resetTime;
    } else if (partNumber[i] == 2) {
      allTiming[i] = cudaTimePart2(timing, vect, size);
      timing = resetTime;
    } else if (partNumber[i] == 3) {
      allTiming[i] = cudaTimePart3(timing, vect, dvect, size);
      timing = resetTime;
    } else if (partNumber[i] == 4) {
      allTiming[i] = cudaTimePart4(timing, vect, dvect, size);
      timing = resetTime;
    } else if (partNumber[i] == 5) {
      allTiming[i] = cudaTimePart5(timing, vect, dvect, size);
      timing = resetTime;
    } else {
      std::cout << "\n\n\tError the User has not chose any number of Function!\n";
      break;
    }
  }

  printoutAll(allTiming, printStander);
  return 0;
}

const std::vector<int> chooseFunction(int toInteger) {
  std::vector<int> digits(0, 0);
  std::vector<int> ERROR(0, 0);

  int digit{1};

  while (toInteger > 0) {
    digit = toInteger % 10;
    if (digit > part) {
      std::cout << "\n\tError Must be integer Argument <= " << part << std::endl;
      return ERROR;
    }
    digits.push_back(digit);
    toInteger /= 10;
  }
  std::reverse(digits.begin(), digits.end());
  return digits;
}

void randomGenerator(std::vector<float> &vect) {
  std::random_device rand;
  std::default_random_engine gener(rand());
  std::uniform_real_distribution<> dis(0., 1.);
  int size = vect.size();
  for (int i = 0; i < size; i++) {
    vect.at(i) = dis(gener);
  }
}
void addVectorsCpu(std::vector<float> &vect1, std::vector<float> &vect2, std::vector<float> &vect3) {
  for (unsigned int i = 0; i < vect1.size(); ++i) {
    vect3[i] = vect2[i] + vect1[i];
  }
}

bool checkingResultsPrintout(std::vector<float> &vectCpu, std::vector<float> &vectGpu) {
  float percent{0.0};
  float totalError{0.0};
  int size = vectCpu.size();
  for (int j = 0; j < size; j++) {
    percent = ((vectCpu[j] - vectGpu[j]) / vectCpu[j]) * 100;
    totalError += percent;
  }
  if (totalError) {
    std::cout << "\n------------------------------------\n";
    std::cout << "| CpuSum | GpuSum | Error  | Error %| ";
    std::cout << "\n------------------------------------\n";
    //std::cout.precision(4);
    for (int j = 0; j < size; j++) {
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
void calculateTimeDuration(Timing &timing, int i) {
  timing.timeCopyToDevice[i] = (timing.copyToDevice[1] - timing.copyToDevice[0]);  //getting the time in microseconds
  timing.timeOperationOnDeviceByHost[i] = (timing.operationOnDeviceByHost[1] - timing.operationOnDeviceByHost[0]);
  timing.timeCopyToHost[i] = (timing.copyToHost[1] - timing.copyToHost[0]);
  cudaEventElapsedTime(&timing.operationOnDeviceByDevice,
                       timing.start,
                       timing.stop);  //get the time elapse in Device operation with device perspective.
  timing.operationOnDeviceByDeviceAcc[i] = (timing.operationOnDeviceByDevice * 1000);
}
void calculateAverageDeviation(Timing &timing) {
  //Average
  for (int i = 0; i < averg; ++i) {
    timing.timeCopyToDevice[averg] += timing.timeCopyToDevice[i];
    timing.timeOperationOnDeviceByHost[averg] += timing.timeOperationOnDeviceByHost[i];
    timing.timeCopyToHost[averg] += timing.timeCopyToHost[i];
    timing.operationOnDeviceByDeviceAcc[averg] += timing.operationOnDeviceByDeviceAcc[i];
  }
  timing.timeCopyToDevice[averg] = timing.timeCopyToDevice[averg] / averg;
  timing.timeOperationOnDeviceByHost[averg] = timing.timeOperationOnDeviceByHost[averg] / averg;
  timing.timeCopyToHost[averg] = timing.timeCopyToHost[averg] / averg;
  timing.operationOnDeviceByDeviceAcc[averg] = (double)timing.operationOnDeviceByDeviceAcc[averg] / averg;

  //Standard deviation
  for (int i = 0; i < averg; ++i) {
    timing.timeCopyToDevice[i] -= timing.timeCopyToDevice[averg];                                  //Take the different.
    timing.timeCopyToDevice[i] = timing.timeCopyToDevice[i] * timing.timeCopyToDevice[i].count();  // Square it.
    timing.timeCopyToDevice[averg + 1] +=
        timing.timeCopyToDevice[i];  //add them togather. averg+1 is location of the Deviation

    timing.timeOperationOnDeviceByHost[i] -= timing.timeOperationOnDeviceByHost[averg];
    timing.timeOperationOnDeviceByHost[i] *= timing.timeOperationOnDeviceByHost[i].count();
    timing.timeOperationOnDeviceByHost[averg + 1] += timing.timeOperationOnDeviceByHost[i];

    timing.timeCopyToHost[i] -= timing.timeCopyToHost[averg];
    timing.timeCopyToHost[i] *= timing.timeCopyToHost[i].count();
    timing.timeCopyToHost[averg + 1] += timing.timeCopyToHost[i];

    timing.operationOnDeviceByDeviceAcc[i] -= timing.operationOnDeviceByDeviceAcc[averg];
    timing.operationOnDeviceByDeviceAcc[i] *= timing.operationOnDeviceByDeviceAcc[i];
    timing.operationOnDeviceByDeviceAcc[averg + 1] += timing.operationOnDeviceByDeviceAcc[i];
  }

  timing.timeCopyToDevice[averg + 1] = timing.timeCopyToDevice[averg + 1] / averg;
  timing.timeCopyToDevice[averg + 1] =
      (std::chrono::duration<double, std::micro>)sqrt(timing.timeCopyToDevice[averg + 1].count());
  timing.timeOperationOnDeviceByHost[averg + 1] = timing.timeOperationOnDeviceByHost[averg + 1] / averg;
  timing.timeOperationOnDeviceByHost[averg + 1] =
      (std::chrono::duration<double, std::micro>)sqrt(timing.timeOperationOnDeviceByHost[averg + 1].count());
  timing.timeCopyToHost[averg + 1] = timing.timeCopyToHost[averg + 1] / averg;
  timing.timeCopyToHost[averg + 1] =
      (std::chrono::duration<double, std::micro>)sqrt(timing.timeCopyToHost[averg + 1].count());

  timing.operationOnDeviceByDeviceAcc[averg + 1] = (double)timing.operationOnDeviceByDeviceAcc[averg + 1] / averg;
  timing.operationOnDeviceByDeviceAcc[averg + 1] = sqrt(timing.operationOnDeviceByDeviceAcc[averg + 1]);
}

void printoutAll(std::vector<Timing> &timing, bool standerDeviationPrint) {
  const std::string gpuReadCpu = " Duration Time Read from Host To Device       ";
  const std::string timeCpu = " Duration Time operation on Host point View   ";
  const std::string timeGpu = " Duration Time operation on Device point View ";
  const std::string cpuReadGpu = " Duration Time Read from Device To Host       ";

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
  std::cout << "   |";

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
  std::cout.width(gpuReadCpu.size() + 3);
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

  newLineTitle(totalFix, gpuReadCpu);
  printResultEach(timing, 1, standerDeviationPrint);
  newLineTitle(totalFix, timeCpu);
  printResultEach(timing, 2, standerDeviationPrint);
  newLineTitle(totalFix, timeGpu);
  printResultEach(timing, 3, standerDeviationPrint);
  newLineTitle(totalFix, cpuReadGpu);
  printResultEach(timing, 4, standerDeviationPrint);

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
      std::cout << timing[i].timeCopyToDevice[averg].count();
      std::cout << "|";
      if (standerDeviationPrint) {
        std::cout.width(standerDeviationWidth);
        std::cout.fill(' ');
        std::cout << timing[i].timeCopyToDevice[averg + 1].count();
        std::cout << "|";
      }
    } else if (type == 2) {
      std::cout.width(averageTimeWidth);
      std::cout.fill(' ');
      std::cout << timing[i].timeOperationOnDeviceByHost[averg].count();
      std::cout << "|";
      if (standerDeviationPrint) {
        std::cout.width(standerDeviationWidth);
        std::cout.fill(' ');
        std::cout << timing[i].timeOperationOnDeviceByHost[averg + 1].count();
        std::cout << "|";
      }
    } else if (type == 3) {
      std::cout.width(averageTimeWidth);
      std::cout.fill(' ');
      std::cout << timing[i].operationOnDeviceByDeviceAcc[averg];
      std::cout << "|";
      if (standerDeviationPrint) {
        std::cout.width(standerDeviationWidth);
        std::cout.fill(' ');
        std::cout << timing[i].operationOnDeviceByDeviceAcc[averg + 1];
        std::cout << "|";
      }
    } else if (type == 4) {
      std::cout.width(averageTimeWidth);
      std::cout.fill(' ');
      std::cout << timing[i].timeCopyToHost[averg].count();
      std::cout << "|";
      if (standerDeviationPrint) {
        std::cout.width(standerDeviationWidth);
        std::cout.fill(' ');
        std::cout << timing[i].timeCopyToHost[averg + 1].count();
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

  file << sizeOfVector << std::endl;
  file << averg << std::endl;
  file << task << std::endl;
  file << timing.timeCopyToDevice[averg].count() << " " << timing.timeCopyToDevice[averg + 1].count() << std::endl;
  file << timing.timeOperationOnDeviceByHost[averg].count() << " "
       << timing.timeOperationOnDeviceByHost[averg + 1].count() << std::endl;
  file << timing.operationOnDeviceByDeviceAcc[averg] << " " << timing.operationOnDeviceByDeviceAcc[averg + 1]
       << std::endl;
  file << timing.timeCopyToHost[averg].count() << " " << timing.timeCopyToHost[averg + 1].count() << std::endl;

  file.close();
  if (!file.good()) {
    std::cout << "\n*ERROR While Writing The " + name + " file!!" << std::endl;
    return 0;
  }
  return 1;
}

Timing cudaTimePart0(Timing &timing, Vectors &vect, Pointers &dvect, int size) {
  std::cout << "\nCudaMalloc is applied Part 0.\n";
  timing.partChosen = 0;

  //////////// Start Average From Here /////////////////////
  for (int i = 0; i < averg; i++) {
    std::fill(vect.vect3Gpu.begin(), vect.vect3Gpu.end(), 0);  //clear each value of vector's elements

    cudaCheck(cudaEventCreate(&timing.start));  //inialize Event.
    cudaCheck(cudaEventCreate(&timing.stop));

    ////////////////////////// Copy From Host To Device //////////////////////////////////
    timing.copyToDevice[0] = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

    timing.copyToDevice[1] = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

    int threads = 512;                                    //arbitrary number.
    int blocks = (sizeOfVector + threads - 1) / threads;  //get ceiling number of blocks.
    blocks = std::min(blocks, 8);  // Number 8 is least number can be got from lowest Nevedia GPUs.

    ////////////////////////// CAll Device Kernel //////////////////////////////////
    cudaCheck(cudaEventRecord(timing.start));
    timing.operationOnDeviceByHost[0] = std::chrono::steady_clock::now();

    addVectorsGpu<<<blocks, threads>>>(vect.vect1.data(),
                                       vect.vect2.data(),
                                       vect.vect3Gpu.data(),
                                       sizeOfVector,
                                       task);  //call device function to add two vectors and save into vect3Gpu.
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaDeviceSynchronize());
    timing.operationOnDeviceByHost[1] = std::chrono::steady_clock::now();
    cudaCheck(cudaEventRecord(timing.stop));

    ////////////////////////// Copy From Device To Host //////////////////////////////////
    timing.copyToHost[0] = std::chrono::steady_clock::now();

    // cudaMemcpy(vect.vect3Gpu.data(), dvect.dVect3, size, cudaMemcpyDeviceToHost);//copy summing result vector from Device to Host.// Try_Regist(3) delete this

    timing.copyToHost[1] = std::chrono::steady_clock::now();

    calculateTimeDuration(timing, i);

    cudaCheck(cudaEventDestroy(timing.start));
    cudaCheck(cudaEventDestroy(timing.stop));
  }

  //////////////////// End Average //////////////////////
  bool test = 0;
  addVectorsCpu(vect.vect1, vect.vect2, vect.vect3Cpu);  //Host is adding vectors too.
  test = checkingResultsPrintout(vect.vect3Cpu,
                                 vect.vect3Gpu);  //Checking the results, if error then Print out to the user.
  if (test) {
    calculateAverageDeviation(timing);
    if (test && saveFile) {
      test = saveToFile("dataPart0", timing);
      std::cout << "\nThe File is saved successfuly.\n";
    }
  }

  return timing;
}
Timing cudaTimePart1(Timing &timing, Vectors &vect, Pointers &dvect, int size, int startSave) {
  std::cout << "\nCudaMalloc is applied Part 1.\n";
  timing.partChosen = 1;
  cudaCheck(
      cudaMalloc((void **)&dvect.dVect1, size));  //allocate memory space for vector in the global memory of the Device.
  cudaCheck(cudaMalloc((void **)&dvect.dVect2, size));
  cudaCheck(cudaMalloc((void **)&dvect.dVect3, size));

  //////////// Start Average From Here /////////////////////
  for (int i = 0; i < averg; i++) {
    std::fill(vect.vect3Gpu.begin(), vect.vect3Gpu.end(), 0);

    cudaCheck(cudaEventCreate(&timing.start));  //inialize Event.
    cudaCheck(cudaEventCreate(&timing.stop));

    ////////////////////////// Copy From Host To Device //////////////////////////////////
    timing.copyToDevice[0] = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

    cudaCheck(cudaMemcpy(
        dvect.dVect1, vect.vect1.data(), size, cudaMemcpyHostToDevice));  //copy random vector from host to device.
    cudaCheck(cudaMemcpy(dvect.dVect2, vect.vect2.data(), size, cudaMemcpyHostToDevice));

    timing.copyToDevice[1] = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

    int threads = 512;                                    //arbitrary number.
    int blocks = (sizeOfVector + threads - 1) / threads;  //get ceiling number of blocks.
    blocks = std::min(blocks, 8);  // Number 8 is least number can be got from lowest Nevedia GPUs.

    ////////////////////////// CAll Device Kernel //////////////////////////////////
    cudaCheck(cudaEventRecord(timing.start));
    timing.operationOnDeviceByHost[0] = std::chrono::steady_clock::now();

    addVectorsGpu<<<blocks, threads>>>(dvect.dVect1,
                                       dvect.dVect2,
                                       dvect.dVect3,
                                       sizeOfVector,
                                       task);  //call device function to add two vectors and save into vect3Gpu.
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaDeviceSynchronize());

    timing.operationOnDeviceByHost[1] = std::chrono::steady_clock::now();
    cudaCheck(cudaEventRecord(timing.stop));

    ////////////////////////// Copy From Device To Host //////////////////////////////////
    timing.copyToHost[0] = std::chrono::steady_clock::now();

    cudaCheck(cudaMemcpy(
        vect.vect3Gpu.data(),
        dvect.dVect3,
        size,
        cudaMemcpyDeviceToHost));  //copy summing result vector from Device to Host.// Try_Regist(3) delete this

    timing.copyToHost[1] = std::chrono::steady_clock::now();

    calculateTimeDuration(timing, i);

    cudaCheck(cudaEventDestroy(timing.start));
    cudaCheck(cudaEventDestroy(timing.stop));
  }

  //////////////////// End Average //////////////////////
  bool test = 0;
  addVectorsCpu(vect.vect1, vect.vect2, vect.vect3Cpu);  //Host is adding vectors too.
  test = checkingResultsPrintout(vect.vect3Cpu,
                                 vect.vect3Gpu);  //Checking the results, if error then Print out to the user.
  if (test) {
    calculateAverageDeviation(timing);
    if (test && saveFile && startSave > 2) {
      test = saveToFile("dataPart1", timing);
      std::cout << "\nThe File is saved successfuly.\n";
    }
  }
  cudaCheck(cudaFree(dvect.dVect1));
  cudaCheck(cudaFree(dvect.dVect2));
  cudaCheck(cudaFree(dvect.dVect3));

  return timing;
}
Timing cudaTimePart2(Timing &timing, Vectors &vect, int size) {
  std::cout << "\nCudaHostRegister is Part 2.\n";
  timing.partChosen = 2;

  //////////// Start Average From Here /////////////////////
  for (int i = 0; i < averg; i++) {
    std::fill(vect.vect3Gpu.begin(), vect.vect3Gpu.end(), 0);

    cudaCheck(cudaEventCreate(&timing.start));  //inialize Event.
    cudaCheck(cudaEventCreate(&timing.stop));

    timing.copyToDevice[0] = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

    cudaCheck(cudaHostRegister(vect.vect1.data(), size, cudaHostRegisterDefault));
    cudaCheck(cudaHostRegister(vect.vect2.data(), size, cudaHostRegisterDefault));
    cudaCheck(cudaHostRegister(vect.vect3Gpu.data(), size, cudaHostRegisterDefault));

    timing.copyToDevice[1] = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

    int threads = 512;                                    //arbitrary number.
    int blocks = (sizeOfVector + threads - 1) / threads;  //get ceiling number of blocks.
    blocks = std::min(blocks, 8);  // Number 8 is least number can be got from lowest Nevedia GPUs.

    timing.operationOnDeviceByHost[0] = std::chrono::steady_clock::now();
    cudaCheck(cudaEventRecord(timing.start));
    cudaCheck(cudaEventSynchronize(
        timing.start));  //If the cudaEventBlockingSync flag has not been set, then the CPU thread will busy-wait until the event has been completed by the GPU.

    addVectorsGpu<<<blocks, threads>>>(vect.vect1.data(),
                                       vect.vect2.data(),
                                       vect.vect3Gpu.data(),
                                       sizeOfVector,
                                       task);  //call device function to add two vectors and save into vect3Gpu.
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaEventRecord(timing.stop));
    cudaCheck(cudaEventSynchronize(timing.stop));

    timing.operationOnDeviceByHost[1] = std::chrono::steady_clock::now();

    timing.copyToHost[0] = std::chrono::steady_clock::now();

    cudaCheck(cudaHostUnregister(vect.vect1.data()));
    cudaCheck(cudaHostUnregister(vect.vect2.data()));
    cudaCheck(cudaHostUnregister(vect.vect3Gpu.data()));

    timing.copyToHost[1] = std::chrono::steady_clock::now();

    calculateTimeDuration(timing, i);

    cudaCheck(cudaEventDestroy(timing.start));
    cudaCheck(cudaEventDestroy(timing.stop));
  }
  //////////////////// End Average //////////////////////
  bool test = 0;
  addVectorsCpu(vect.vect1, vect.vect2, vect.vect3Cpu);  //Host is adding vectors too.

  test = checkingResultsPrintout(vect.vect3Cpu,
                                 vect.vect3Gpu);  //Checking the results, if error then Print out to the user.

  if (test) {
    calculateAverageDeviation(timing);
    if (test && saveFile) {
      test = saveToFile("dataPart2", timing);
      std::cout << "\nThe File is saved successfuly.\n";
    }
  }

  return timing;
}

Timing cudaTimePart3(Timing &timing, Vectors &vect, Pointers &dvect, int size) {
  std::cout << "\nCudaMallocHost is applied Part 3.\n";
  timing.partChosen = 3;

  cudaCheck(cudaMallocHost((void **)&dvect.dVect1, size));  //allocate memory space for vector in the host memory.
  cudaCheck(cudaMallocHost((void **)&dvect.dVect2, size));
  cudaCheck(cudaMallocHost((void **)&dvect.dVect3, size));

  //////////// Start Average From Here /////////////////////
  for (int i = 0; i < averg; i++) {
    std::fill(vect.vect3Gpu.begin(), vect.vect3Gpu.end(), 0);

    cudaCheck(cudaEventCreate(&timing.start));  //inialize Event.
    cudaCheck(cudaEventCreate(&timing.stop));

    timing.copyToDevice[0] = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

    cudaCheck(cudaMemcpy(dvect.dVect1, vect.vect1.data(), size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dvect.dVect2, vect.vect2.data(), size, cudaMemcpyHostToDevice));

    timing.copyToDevice[1] = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

    int threads = 512;                                    //arbitrary number.
    int blocks = (sizeOfVector + threads - 1) / threads;  //get ceiling number of blocks.
    blocks = std::min(blocks, 8);  // Number 8 is least number can be got from lowest Nevedia GPUs.

    timing.operationOnDeviceByHost[0] = std::chrono::steady_clock::now();
    cudaCheck(cudaEventRecord(timing.start));
    cudaCheck(cudaEventSynchronize(
        timing.start));  //Waits for an event to complete.If the cudaEventBlockingSync flag has not been set, then the CPU thread will busy-wait until the event has been completed by the GPU.

    addVectorsGpu<<<blocks, threads>>>(dvect.dVect1,
                                       dvect.dVect2,
                                       dvect.dVect3,
                                       sizeOfVector,
                                       task);  //call device function to add two vectors and save into vect3Gpu.
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaEventRecord(timing.stop));
    cudaCheck(cudaEventSynchronize(timing.stop));

    timing.operationOnDeviceByHost[1] = std::chrono::steady_clock::now();

    timing.copyToHost[0] = std::chrono::steady_clock::now();

    cudaCheck(cudaMemcpy(vect.vect3Gpu.data(), dvect.dVect3, size, cudaMemcpyDeviceToHost));

    timing.copyToHost[1] = std::chrono::steady_clock::now();

    calculateTimeDuration(timing, i);

    cudaCheck(cudaEventDestroy(timing.start));
    cudaCheck(cudaEventDestroy(timing.stop));
  }
  //////////////////// End Average //////////////////////
  bool test = 0;
  addVectorsCpu(vect.vect1, vect.vect2, vect.vect3Cpu);  //Host is adding vectors too.

  test = checkingResultsPrintout(vect.vect3Cpu,
                                 vect.vect3Gpu);  //Checking the results, if error then Print out to the user.

  if (test) {
    calculateAverageDeviation(timing);
    if (test && saveFile) {
      test = saveToFile("dataPart3", timing);
      std::cout << "\nThe File is saved successfuly.\n";
    }
  }
  cudaCheck(cudaFreeHost(dvect.dVect1));
  cudaCheck(cudaFreeHost(dvect.dVect2));
  cudaCheck(cudaFreeHost(dvect.dVect3));
  return timing;
}
Timing cudaTimePart4(Timing &timing, Vectors &vect, Pointers &dvect, int size) {
  std::cout << "\nCudaMallocHost is applied Part 4\n";
  timing.partChosen = 4;

  //Using cudaMallocHost for pinning Vector Memory.
  cudaCheck(cudaMallocHost((void **)&dvect.dVect1, size));  //allocate memory inside the host and pinned that memory.
  cudaCheck(cudaMallocHost((void **)&dvect.dVect2, size));
  cudaCheck(cudaMallocHost((void **)&dvect.dVect3, size));

  cudaCheck(cudaMalloc((void **)&dvect.dVect1Extra, size));  //Allocate memory inside the device.
  cudaCheck(cudaMalloc((void **)&dvect.dVect2Extra, size));
  cudaCheck(cudaMalloc((void **)&dvect.dVect3Extra, size));

  //////////// Start Average From Here /////////////////////
  for (int i = 0; i < averg; i++) {
    std::fill(vect.vect3Gpu.begin(), vect.vect3Gpu.end(), 0);

    cudaCheck(cudaEventCreate(&timing.start));  //inialize Event.
    cudaCheck(cudaEventCreate(&timing.stop));

    timing.copyToDevice[0] = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

    memcpy(dvect.dVect1, vect.vect1.data(), size);  //Copy from vector host to pinned buffer Host.
    memcpy(dvect.dVect2, vect.vect2.data(), size);

    cudaCheck(cudaMemcpy(dvect.dVect1Extra, dvect.dVect1, size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dvect.dVect2Extra, dvect.dVect2, size, cudaMemcpyHostToDevice));

    timing.copyToDevice[1] = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

    int threads = 512;                                    //arbitrary number.
    int blocks = (sizeOfVector + threads - 1) / threads;  //get ceiling number of blocks.
    blocks = std::min(blocks, 8);  // Number 8 is least number can be got from lowest Nevedia GPUs.

    timing.operationOnDeviceByHost[0] = std::chrono::steady_clock::now();
    cudaCheck(cudaEventRecord(timing.start));
    cudaCheck(cudaEventSynchronize(
        timing.start));  //Waits for an event to complete.If the cudaEventBlockingSync flag has not been set, then the CPU thread will busy-wait until the event has been completed by the GPU.

    addVectorsGpu<<<blocks, threads>>>(dvect.dVect1Extra,
                                       dvect.dVect2Extra,
                                       dvect.dVect3Extra,
                                       sizeOfVector,
                                       task);  //call device function to add two vectors and save into vect3Gpu.
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaEventRecord(timing.stop));
    cudaCheck(cudaEventSynchronize(timing.stop));

    timing.operationOnDeviceByHost[1] = std::chrono::steady_clock::now();

    timing.copyToHost[0] = std::chrono::steady_clock::now();

    cudaCheck(cudaMemcpy(dvect.dVect3, dvect.dVect3Extra, size, cudaMemcpyDeviceToHost));
    memcpy(vect.vect3Gpu.data(), dvect.dVect3, size);  //copy pinned host buffer to vector host.

    timing.copyToHost[1] = std::chrono::steady_clock::now();

    calculateTimeDuration(timing, i);

    cudaCheck(cudaEventDestroy(timing.start));
    cudaCheck(cudaEventDestroy(timing.stop));
  }
  //////////////////// End Average //////////////////////
  bool test = 0;
  addVectorsCpu(vect.vect1, vect.vect2, vect.vect3Cpu);  //Host is adding vectors too.

  test = checkingResultsPrintout(vect.vect3Cpu,
                                 vect.vect3Gpu);  //Checking the results, if error then Print out to the user.
  if (test) {
    calculateAverageDeviation(timing);
    if (test && saveFile) {
      test = saveToFile("dataPart4", timing);
      std::cout << "\nThe File is saved successfuly.\n";
    }
  }

  cudaCheck(cudaFreeHost(dvect.dVect1));
  cudaCheck(cudaFreeHost(dvect.dVect2));
  cudaCheck(cudaFreeHost(dvect.dVect3));
  cudaCheck(cudaFree(dvect.dVect1Extra));
  cudaCheck(cudaFree(dvect.dVect2Extra));
  cudaCheck(cudaFree(dvect.dVect3Extra));

  return timing;
}

Timing cudaTimePart5(Timing &timing, Vectors &vect, Pointers &dvect, int size) {
  std::cout << "\nCudaHostRegister is applied Part 5.\n";
  timing.partChosen = 5;

  cudaCheck(
      cudaMalloc((void **)&dvect.dVect1, size));  //allocate memory space for vector in the global memory of the Device.
  cudaCheck(cudaMalloc((void **)&dvect.dVect2, size));
  cudaCheck(cudaMalloc((void **)&dvect.dVect3, size));

  //////////// Start Average From Here /////////////////////
  for (int i = 0; i < averg; i++) {
    std::fill(vect.vect3Gpu.begin(), vect.vect3Gpu.end(), 0);

    cudaCheck(cudaEventCreate(&timing.start));  //inialize Event.
    cudaCheck(cudaEventCreate(&timing.stop));

    timing.copyToDevice[0] = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
    cudaCheck(cudaHostRegister(vect.vect1.data(), size, cudaHostRegisterDefault));
    cudaCheck(cudaHostRegister(vect.vect2.data(), size, cudaHostRegisterDefault));
    cudaCheck(cudaHostRegister(vect.vect3Gpu.data(), size, cudaHostRegisterDefault));

    cudaCheck(cudaMemcpy(dvect.dVect1,
                         vect.vect1.data(),
                         size,
                         cudaMemcpyHostToDevice));  //copy pinned vector in the host to buffer in the device.
    cudaCheck(cudaMemcpy(dvect.dVect2, vect.vect2.data(), size, cudaMemcpyHostToDevice));

    timing.copyToDevice[1] = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

    int threads = 512;                                    //arbitrary number.
    int blocks = (sizeOfVector + threads - 1) / threads;  //get ceiling number of blocks.
    blocks = std::min(blocks, 8);  // Number 8 is least number can be got from lowest Nevedia GPUs.

    timing.operationOnDeviceByHost[0] = std::chrono::steady_clock::now();
    cudaCheck(cudaEventRecord(timing.start));
    cudaCheck(cudaEventSynchronize(
        timing.start));  //If the cudaEventBlockingSync flag has not been set, then the CPU thread will busy-wait until the event has been completed by the GPU.

    addVectorsGpu<<<blocks, threads>>>(dvect.dVect1,
                                       dvect.dVect2,
                                       dvect.dVect3,
                                       sizeOfVector,
                                       task);  //call device function to add two vectors and save into vect3Gpu.
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaEventRecord(timing.stop));
    cudaCheck(cudaEventSynchronize(timing.stop));

    timing.operationOnDeviceByHost[1] = std::chrono::steady_clock::now();

    timing.copyToHost[0] = std::chrono::steady_clock::now();

    cudaCheck(cudaMemcpy(vect.vect3Gpu.data(),
                         dvect.dVect3,
                         size,
                         cudaMemcpyDeviceToHost));  //copy  buffer in the device to pinned vector in the host.
    cudaCheck(cudaHostUnregister(vect.vect1.data()));
    cudaCheck(cudaHostUnregister(vect.vect2.data()));
    cudaCheck(cudaHostUnregister(vect.vect3Gpu.data()));

    timing.copyToHost[1] = std::chrono::steady_clock::now();

    calculateTimeDuration(timing, i);

    cudaCheck(cudaEventDestroy(timing.start));
    cudaCheck(cudaEventDestroy(timing.stop));
  }
  //////////////////// End Average //////////////////////
  bool test = 0;
  addVectorsCpu(vect.vect1, vect.vect2, vect.vect3Cpu);  //Host is adding vectors too.

  test = checkingResultsPrintout(vect.vect3Cpu,
                                 vect.vect3Gpu);  //Checking the results, if error then Print out to the user.

  if (test) {
    calculateAverageDeviation(timing);
    if (test && saveFile) {
      test = saveToFile("dataPart5", timing);
      std::cout << "\nThe File is saved successfuly.\n";
    }
  }
  cudaCheck(cudaFree(dvect.dVect1));
  cudaCheck(cudaFree(dvect.dVect2));
  cudaCheck(cudaFree(dvect.dVect3));
  return timing;
}
