def check_torch_gpu(torch, device):
    print("GPU available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print("Device name:", torch.cuda.get_device_name(i))
    print("HIP version:", torch.version.hip)
    print("CUDA version:", torch.version.cuda)
    gpu_device = "cuda"
    gpu = True
    if device == "cuda":
        print("CUDA version:", torch.version.cuda)
        if not torch.version.cuda:
            print("CUDA device not found")
            gpu = False
    elif device == "rocm":
        print("HIP version:", torch.version.hip)
        if not torch.version.hip:
            print("ROCM device not found")
            gpu = False
    else:
        gpu_device = "cpu"
    gpu_name = "CPU"
    if gpu_device != "cpu":
        gpu_name = torch.cuda.get_device_name(0)
    print("Torch Device:", gpu, gpu_device, gpu_name)
    return (gpu, gpu_device, gpu_name)
