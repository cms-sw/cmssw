#include <stdio.h>
#include <math.h>

int main(void)
{
  double y, dy, y0, dy0;
  double chi;
  
  scanf("%lE%lE%lE%lE", &y, &dy, &y0, &dy0);
  
  if (dy == 0. && dy0 == 0.)
    chi = -1.;
  else
    chi = (y - y0) / sqrt(dy*dy + dy0*dy0);
  
  printf("%lE %lE  %lE %lE  %lf ", y, dy, y0, dy0, chi);
  
  if ( (fabs(y0) < 5*dy0 || fabs(y) <= 4*dy) && fabs(chi) <= 3.)
    printf("[BADSTAT]\n");
  else
    if (fabs(chi) > 3. || fabs(chi) <= 3.) { // this if is to deal with NaN
      if (fabs(chi) > 3.) printf("[DEVIATION]\n");
      if (fabs(chi) <= 3.) printf("[OK]\n");
    }
    else
      printf("[DEVIATION]\n");
  
  return 0;
}
