from collections import namedtuple
from .offline_config import LAYOUTS as offline_layouts
from .online_config import LAYOUTS as online_layouts

def register_layout(plot):
    LayoutManager.add_layout(plot)


class LayoutManager:
  __layouts = []
  __offline_layouts = []
  __online_layouts = []

  @classmethod
  def get_layouts(cls, dataset):
    if dataset == '/Global/Online/ALL':
      return cls.__online_layouts
    else:
      return cls.__offline_layouts

  @classmethod
  def add_layout(cls, layout_object):
    keys = list(layout_object.keys())
    Layout = namedtuple('Layout', keys, defaults=['default'])
    values = []
    for key in keys:
        values.append(layout_object[key])
    layout = Layout._make(values)
    if not layout or not layout.source or not layout.destination or not layout.name:
        raise Exception('source, destination and name has to be provided for the layout.')
    if layout_object['file_path'] in online_layouts:
        cls.__online_layouts.append(layout)
    elif layout_object['file_path'] in offline_layouts:
        cls.__offline_layouts.append(layout)
    elif layout_object['destination'].split('/')[0] == 'Summary':
        cls.__offline_layouts.append(layout)
        cls.__online_layouts.append(layout)
    cls.__layouts.append(layout)


  @classmethod
  def get_layouts_by_name(cls, name):
    if not name:
      return []
    return [x for x in cls.__layouts if x.name == name]
