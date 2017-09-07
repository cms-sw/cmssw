/*
 * TensorFlow input/output interface.
 * Based on TensorFlow C API 1.1.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include "PhysicsTools/TensorFlow/interface/IO.h"

namespace tf
{

IO::IO(Tensor* tensor, TF_Operation* tf_operation, const std::string& opName, int opIndex)
    : tensor_(tensor)
    , opName_(opName)
{
    // create the tf_output object, i.e., a struct of op and index
    tf_output_.oper = tf_operation;
    tf_output_.index = opIndex;
}

IO::~IO()
{
}

} // namespace tf
