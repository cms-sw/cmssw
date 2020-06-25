from collections import namedtuple


def register_layout(source, destination, name='default'):
    LayoutManager.add_layout(Layout(source, destination, name))


class LayoutManager:
    __layouts = []

    @classmethod
    def get_layouts(cls):
        return cls.__layouts


    @classmethod
    def add_layout(cls, layout):
        if not layout or not layout.source or not layout.destination or not layout.name:
            raise Exception('source, destination and name has to be provided for the layout.')
        cls.__layouts.append(layout)


    @classmethod
    def get_layouts_by_name(cls, name):
        if not name:
            return []
        return [x for x in cls.__layouts if x.name == name]


# Name has a default value
Layout = namedtuple('Layout', ['source', 'destination', 'name'], defaults=['default'])
