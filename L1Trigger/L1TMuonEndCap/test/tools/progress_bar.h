#ifndef PAPERINO_PROGRESS_BAR_HH
#define PAPERINO_PROGRESS_BAR_HH

#include <cstdio>
#include <ctime>
#include <iostream>
#include <iomanip>

namespace {

  // Inspired by Olmo Cerri (SNS Pisa) and Boost C++ progress_display
  double elapsed_time() {
    static std::clock_t start_time = std::clock();
    return (static_cast<double>(std::clock() - start_time)/CLOCKS_PER_SEC);
  }

  void show_progress_bar(unsigned long entry, unsigned long num_entries) {
    static double progress;
    static unsigned int num_steps=100, progress_step=0, next_progress_step=0;
    static unsigned int time_left=0, hr_left=0, min_left=0;

    num_entries = (num_entries == 0) ? 1 : num_entries;  // protect against 0

    if (entry == 0) {
      //std::cout << std::endl;

      elapsed_time();  // start the timer
    }

    progress = static_cast<double>(num_steps)*(entry+1)/num_entries;
    progress_step = progress;

    if (progress_step >= next_progress_step) {
      unsigned int i = 0;
      std::cout << "\r" << "[";
      for (; i<next_progress_step; i+=5) {
        std::cout << "#";
      }
      for (; i<num_steps; i+=5) {
        std::cout << "-";
      }
      std::cout << "]  " << progress_step*100/num_steps << "%";

      if (progress_step < num_steps) {
        time_left = elapsed_time() / (entry+1) * (num_entries - entry);
        hr_left = time_left/3600;
        min_left = (time_left/60) - (hr_left*60);
        time_left = time_left - (min_left*60) - (hr_left*3600);

        std::cout << "  approx "  << std::setw(2) << hr_left << "h " << std::setw(2) << min_left << "m "  << std::setw(2) << time_left << "s remaining";
      } else {
        std::cout << "                             ";  // clear line
      }
      std::cout.flush();

      next_progress_step += 1;
    }

    if (entry == num_entries-1) {
      std::cout << std::endl;  // new line and flush

      std::cout << "Elapsed time: " << elapsed_time() << " sec" << std::endl;
    }
  }

}  // namespace

#endif
