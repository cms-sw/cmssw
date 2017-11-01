#ifndef _CUHOOK_H_
# define _CUHOOK_H_

typedef enum HookTypesEnum {
    PRE_CALL_HOOK,
    POST_CALL_HOOK,
    CU_HOOK_TYPES,
} HookTypes;

typedef enum HookSymbolsEnum {
    CU_HOOK_MEM_ALLOC,
    CU_HOOK_MEM_FREE,
    CU_HOOK_CTX_GET_CURRENT,
    CU_HOOK_CTX_SET_CURRENT,
    CU_HOOK_CTX_DESTROY,
    CU_HOOK_SYMBOLS,
} HookSymbols;

// One and only function to call to register a callback
// You need to dlsym this symbol in your application and call it to register callbacks
typedef void (*fnCuHookRegisterCallback)(HookSymbols symbol, HookTypes type, void* callback);
extern "C" { void cuHookRegisterCallback(HookSymbols symbol, HookTypes type, void* callback); }

// In case you want to intercept, the callbacks need the same type/parameters as the real functions
typedef CUresult CUDAAPI (*fnMemAlloc)(CUdeviceptr *dptr, size_t bytesize);
typedef CUresult CUDAAPI (*fnMemFree)(CUdeviceptr dptr);
typedef CUresult CUDAAPI (*fnCtxGetCurrent)(CUcontext *pctx);
typedef CUresult CUDAAPI (*fnCtxSetCurrent)(CUcontext ctx);
typedef CUresult CUDAAPI (*fnCtxDestroy)(CUcontext ctx);

#endif /* _CUHOOK_H_ */
