def makeSerialClone(module, **kwargs):
    type = module._TypedParameterizable__type
    if type.endswith('@alpaka'):
        # alpaka module with automatic backend selection
        base = type.removesuffix('@alpaka')
    elif type.startswith('alpaka_serial_sync::'):
        # alpaka module with explicit serial_sync backend
        base = type.removeprefix('alpaka_serial_sync::')
    elif type.startswith('alpaka_cuda_async::'):
        # alpaka module with explicit cuda_async backend
        base = type.removeprefix('alpaka_cuda_async::')
    elif type.startswith('alpaka_rocm_async::'):
        # alpaka module with explicit rocm_async backend
        base = type.removeprefix('alpaka_rocm_async::')
    else:
        # non-alpaka module
        raise TypeError('%s is not an alpaka-based module, and cannot be used with makeSerialClone()' % str(module))

    copy = module.clone(**kwargs)
    copy._TypedParameterizable__type = 'alpaka_serial_sync::' + base
    if 'alpaka' in copy.parameterNames_():
        del copy.alpaka
    return copy
