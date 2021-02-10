#ifndef PLATFORMDEPENDANTOPTIONS
#define PLATFORMDEPENDANTOPTIONS

#ifdef WIN32
#pragma warning(disable:4290)  // disable throw specs warnings introduced in VC++ .NET 2003.
#ifdef PLATFORMDEPENDANT_STATIC
#define PLATFORMDEPENDANT_DLL_API
#else
//#include "StdAfx.h"
#ifdef PLATFORMDEPENDANT_DLL_EXPORTS
#define PLATFORMDEPENDANT_DLL_API __declspec(dllexport)
#else
#define PLATFORMDEPENDANT_DLL_API __declspec(dllimport)
#endif // PLATFORMDEPENDANT_DLL_EXPORTS
#endif // PLATFORMDEPENDANT_STATIC
#else
#define PLATFORMDEPENDANT_DLL_API
#endif // WIN32

#endif // PLATFORMDEPENDANTOPTIONS

