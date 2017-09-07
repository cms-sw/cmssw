/*
 * TensorFlow input/output interface.
 * Based on TensorFlow C API 1.1.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#ifndef PHYSICSTOOLS_TENSORFLOW_IO_H
#define PHYSICSTOOLS_TENSORFLOW_IO_H

#include <vector>

#include <tensorflow/c/c_api.h>

#include "PhysicsTools/TensorFlow/interface/Tensor.h"

namespace tf
{

// generic class containing all information of inputs to / outputs from a graph
class IO
{
public:
    // constructur
    IO(Tensor* tensor, TF_Operation* tf_operation, const std::string& opName, int opIndex = 0);

    // disable implicit copy constructor
    IO(const IO&) = delete;

    // destructor
    ~IO();

    // returns the pointer to the tensor instance
    inline Tensor* getTensor()
    {
        return tensor_;
    }

    // returns a reference to the tensorflow output object
    inline TF_Output& getTFOutput()
    {
        return tf_output_;
    }

    // returns the operation name
    inline std::string& getOpName()
    {
        return opName_;
    }

    // returns the operation index
    inline int getOpIndex()
    {
        return tf_output_.index;
    }

private:
    Tensor* tensor_;
    std::string opName_;

    TF_Output tf_output_;
};

// typedefs
typedef std::vector<IO*> IOs;

} // namepace tf

#endif // PHYSICSTOOLS_TENSORFLOW_IO_H
