#ifndef FWCore_AbstractServices_ResourceInformation_h
#define FWCore_AbstractServices_ResourceInformation_h

/** \class edm::ResourceInformation

  Description:

  Usage:

\author W. David Dagenhart, created May 3, 2022
*/

#include <string>
#include <vector>

namespace edm {

  class ResourceInformation {
  public:
    ResourceInformation();
    ResourceInformation(ResourceInformation const&) = delete;
    ResourceInformation const& operator=(ResourceInformation const&) = delete;
    virtual ~ResourceInformation();

    virtual std::vector<std::string> const& selectedAccelerators() const = 0;
    virtual std::vector<std::string> const& cpuModels() const = 0;
    virtual std::vector<std::string> const& gpuModels() const = 0;

    virtual bool hasGpuNvidia() const = 0;

    virtual std::string const& nvidiaDriverVersion() const = 0;
    virtual int cudaDriverVersion() const = 0;
    virtual int cudaRuntimeVersion() const = 0;

    // Same as cpuModels except in a single string with models separated by ", "
    virtual std::string const& cpuModelsFormatted() const = 0;
    virtual double cpuAverageSpeed() const = 0;

    virtual void setSelectedAccelerators(std::vector<std::string> const& selectedAccelerators) = 0;
    virtual void setCPUModels(std::vector<std::string> const&) = 0;
    virtual void setGPUModels(std::vector<std::string> const&) = 0;

    virtual void setNvidiaDriverVersion(std::string const&) = 0;
    virtual void setCudaDriverVersion(int) = 0;
    virtual void setCudaRuntimeVersion(int) = 0;

    virtual void setCpuModelsFormatted(std::string const&) = 0;
    virtual void setCpuAverageSpeed(double) = 0;
  };
}  // namespace edm
#endif
