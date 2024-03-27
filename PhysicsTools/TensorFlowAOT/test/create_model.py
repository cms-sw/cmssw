# coding: utf-8

"""
Test script to create a simple model for AOT compilation.

By default, a simple float32 -> float32 is created. When "--multi-tensor" is defined, the signature
is (float32, float64) -> (float32, bool).
"""

import os

import cmsml


def create_model(model_dir, multi_tensor=False):
    # get tensorflow (suppressing the usual device warnings and logs)
    tf = cmsml.tensorflow.import_tf()[0]

    # set random seeds to get deterministic results for testing
    tf.keras.utils.set_random_seed(1)

    # define architecture
    n_in, n_out, n_layers, n_units = 4, 2, 5, 128

    # define input layer(s)
    if multi_tensor:
        x1 = tf.keras.Input(shape=(n_in,), dtype=tf.float32, name="input1")
        x2 = tf.keras.Input(shape=(n_in,), dtype=tf.float64, name="input2")
        x = tf.keras.layers.Concatenate(axis=1)([x1, x2])
    else:
        x1 = tf.keras.Input(shape=(n_in,), dtype=tf.float32, name="input1")
        x = x1

    # model layers
    a = tf.keras.layers.BatchNormalization(axis=1, renorm=True)(x)
    for _ in range(n_layers):
        a = tf.keras.layers.Dense(n_units, activation="tanh")(a)
        a = tf.keras.layers.BatchNormalization(axis=1, renorm=True)(a)
    y1 = tf.keras.layers.Dense(n_out, activation="softmax", name="output1", dtype=tf.float32)(a)

    # define output layer(s)
    if multi_tensor:
        y2 = tf.keras.layers.Reshape((n_out,), name="output2")(y1 > 0.5)

    # define the model
    inputs, outputs = [x1], [y1]
    if multi_tensor:
        inputs.append(x2)
        outputs.append(y2)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # test evaluation
    inputs = [
        tf.constant([list(range(n_in))], dtype=tf.float32),
    ]
    if multi_tensor:
        inputs.append(tf.constant([list(range(n_in))], dtype=tf.float64))
    print(model(inputs))

    # save it
    tf.saved_model.save(model, model_dir)


def main():
    from argparse import ArgumentParser

    this_dir = os.path.dirname(os.path.abspath(__file__))
    aot_dir = os.path.dirname(this_dir)

    parser = ArgumentParser(
        description="create a simple model for AOT compilation",
    )
    parser.add_argument(
        "--model-dir",
        "-d",
        default=os.path.join(aot_dir, "data", "testmodel"),
        help="the model directory; default: %(default)s",
    )
    parser.add_argument(
        "--multi-tensor",
        "-m",
        action="store_true",
        help="create a model with multiple inputs and outputs",
    )
    args = parser.parse_args()
    create_model(args.model_dir, multi_tensor=args.multi_tensor)


if __name__ == "__main__":
    main()
