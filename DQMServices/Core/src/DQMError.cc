#include "DQMServices/Core/src/DQMError.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#if WITHOUT_CMS_FRAMEWORK
static const char FAILED[] = "(out of memory while formatting error message)";
#endif

void
raiseDQMError(const char *context, const char *fmt, ...)
{
  va_list args;
  char *message = 0;

  va_start(args, fmt);
  vasprintf(&message, fmt, args);
  va_end(args);

#if WITHOUT_CMS_FRAMEWORK
  char *final = 0;
  asprintf(&final, "%s: %s", context, message ? message : FAILED);
  std::runtime_error err(final ? final : FAILED);
  free(final);
#else
  cms::Exception err(context);
  if (message)
    err << message;
#endif

  free(message);
  throw err;
}
