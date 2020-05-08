
def register_layout(source, destination, name='default'):
    LayoutManager.add_layout(Layout(source, destination, name))


class LayoutManager:
    __layouts = []

    @classmethod
    def get_layouts(cls):
        return cls.__layouts


    @classmethod
    def add_layout(cls, layout):
        cls.__layouts.append(layout)


class Layout:
    def __init__(self, source, destination, name='default'):
        if not source or not destination or not name:
            raise Exception('source, destination and name has to be provided for the layout.')

        self.source = source
        self.destination = destination
        self.name = name