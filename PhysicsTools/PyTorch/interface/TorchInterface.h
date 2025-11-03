#ifndef PhysicsTools_PyTorch_interface_TorchInterface_h
#define PhysicsTools_PyTorch_interface_TorchInterface_h

// TorchLib and ROOT's ClassDef macro clash.
// See: https://root-forum.cern.ch/t/use-of-torch-model-inside-root-dataframe-class-functor/62797
#pragma push_macro("ClassDef")
#undef ClassDef
#include <torch/torch.h>
#include <torch/script.h>
#pragma pop_macro("ClassDef")

#endif  // PhysicsTools_PyTorch_interface_TorchInterface_h
