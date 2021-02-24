from collections import namedtuple
from ..layout_manager import register_layout

def apply_dqm_items_to_new_back_end(dqmitems, file_path):
  layouts_paths = list(dqmitems.keys())
  for layout_path in layouts_paths:
    path_and_layout_name = layout_path.split('/')
    layout_name = path_and_layout_name.pop()
    destination = ('/').join(path_and_layout_name)
    lines_amount_in_layout = len(dqmitems[layout_path])
    plots = []

    for n in range(lines_amount_in_layout):
      plots_in_one_row = dqmitems[layout_path][n] # In old backend, plots are made in tuple and one element of tuple is array. Array reflects a line in html table
      for one_plot in plots_in_one_row:
        if one_plot is not None:
            if ('path' in one_plot):
                plot_name = one_plot['path'].split('/').pop()
                one_plot['source'] = one_plot.pop('path')
                one_plot['destination'] = ('/').join([destination, plot_name])
                one_plot['name'] = layout_name
                one_plot['file_path'] = file_path
                register_layout(one_plot)

            else:
                parts = one_plot.split('/')
                plot_name = parts.pop()
                path = ('/').join(parts)
                plot_destination =  ('/').join([destination, plot_name])
                plot = {'source': one_plot, 'destination': plot_destination, 'name': layout_name, 'file_path': file_path }
                register_layout(plot)
